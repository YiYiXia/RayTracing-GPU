// smallptCUDA by Sam Lapere, 2015
// based on smallpt, a path tracer by Kevin Beason, 2008  
 
#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.h" // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include<opencv2/opencv.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include"Object.h"
#include"Math.h"

#define M_PI 3.14159265359f  // pi
#define width 1024  // screenwidth
#define height 768 // screenheight
#define samps 1 // samples 

//typedef texture<float4, 1, cudaReadModeElementType> triangleTexture;
// __device__ : executed on the device (GPU) and callable only from the device
int total_number_of_triangles = 0;
\
float3 scene_aabbox_min;
float3 scene_aabbox_max;
int frames = 0;
// AXIS ALIGNED BOXES
texture<float4, 1, cudaReadModeElementType> triangle_texture;
// helper functions



__device__ float RayTriangleIntersection(const Ray &r,
	const float3 &v0,
	const float3 &edge1,
	const float3 &edge2)
{

	float3 tvec = r.orig - v0;
	float3 pvec = cross(r.dir, edge2);
	float  det = dot(edge1, pvec);

	det = __fdividef(1.0f, det);  // CUDA intrinsic function 
	//det = 1 / det;							  //det = 1 / det;
	float u = dot(tvec, pvec) * det;

	if (u < 0.0f || u > 1.0f)
		return -1.0f;

	float3 qvec = cross(tvec, edge1);

	float v = dot(r.dir, qvec) * det;

	if (v < 0.0f || (u + v) > 1.0f)
		return -1.0f;

	return dot(edge2, qvec) * det;
}

__device__ float3 getTriangleNormal(const int triangleIndex) {

	float4 edge1 = tex1Dfetch(triangle_texture, triangleIndex * 3 + 1);
	float4 edge2 = tex1Dfetch(triangle_texture, triangleIndex * 3 + 2);

	// cross product of two triangle edges yields a vector orthogonal to triangle plane
	float3 trinormal = cross(make_float3(edge1.x, edge1.y, edge1.z), make_float3(edge2.x, edge2.y, edge2.z));
	trinormal = normalize(trinormal);

	return trinormal;
}

__device__ void intersectAllTriangles(const Ray& r, float& t_scene, int& triangle_id, const int number_of_triangles, int& geomtype) {

	for (int i = 0; i < number_of_triangles; i++)
	{
		// the triangles are packed into the 1D texture using three consecutive float4 structs for each triangle, 
		// first float4 contains the first vertex, second float4 contains the first precomputed edge, third float4 contains second precomputed edge like this: 
		// (float4(vertex.x,vertex.y,vertex.z, 0), float4 (egde1.x,egde1.y,egde1.z,0),float4 (egde2.x,egde2.y,egde2.z,0)) 

		// i is triangle index, each triangle represented by 3 float4s in triangle_texture
		float4 v0 = tex1Dfetch(triangle_texture, i * 3);
		float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
		float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

		// intersect ray with reconstructed triangle	
		float t = RayTriangleIntersection(r,
			make_float3(v0.x, v0.y, v0.z),
			make_float3(edge1.x, edge1.y, edge1.z),
			make_float3(edge2.x, edge2.y, edge2.z));

		// keep track of closest distance and closest triangle
		// if ray/tri intersection finds an intersection point that is closer than closest intersection found so far
		if (t < t_scene && t > 0.001)
		{
			t_scene = t;
			triangle_id = i;
			geomtype = 3;
		}
	}
}

// load triangle data in a CUDA texture
//extern "C"
//{
//	
//}
void bindTriangles(float *dev_triangle_p, unsigned int number_of_triangles, texture<float4, 1, cudaReadModeElementType> &texture)
{
	//triangle_texture.normalized = false;                      // access with normalized texture coordinates
	//triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	//triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates
	texture.normalized = false;                      // access with normalized texture coordinates
	texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

	size_t size = sizeof(float4)*number_of_triangles * 3;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaBindTexture(0, triangle_texture, dev_triangle_p, channelDesc, size);
}




TriangleMesh mesh1;

float *dev_triangle_p; // the cuda device pointer that points to the uploaded triangles


void initCUDAmemoryTriMesh()
{
	loadObj("bunny.obj", mesh1);

	//for (int i = 0; i < mesh1.verts.size(); i++)
	//{

	//}

	// scalefactor and offset to position/scale triangle meshes
	float scalefactor1 = 300;
	float3 offset1 = make_float3(0.15, 0.03, 0.1);// (30, -2, 80);

	std::vector<float4> triangles;

	for (unsigned int i = 0; i < mesh1.faces.size(); i++)
	{
		// make a local copy of the triangle vertices
		float3 v0 = mesh1.verts[mesh1.faces[i].v[0] - 1];
		float3 v1 = mesh1.verts[mesh1.faces[i].v[1] - 1];
		float3 v2 = mesh1.verts[mesh1.faces[i].v[2] - 1];

		

		// translate
		v0 += offset1;
		v1 += offset1;
		v2 += offset1;

		// scale
		v0 *= scalefactor1;
		v1 *= scalefactor1;
		v2 *= scalefactor1;
		// store triangle data as float4
		// store two edges per triangle instead of vertices, to save some calculations in the
		// ray triangle intersection test
		triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
		triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
	}

	// compute bounding box of this mesh
	mesh1.bounding_box[0] += offset1; mesh1.bounding_box[0] *= scalefactor1;
	mesh1.bounding_box[1] += offset1; mesh1.bounding_box[1] *= scalefactor1;

	std::cout << "total number of triangles check:" << mesh1.faces.size() << " == " << triangles.size() / 3 << std::endl;

	// calculate total number of triangles in the scene
	size_t triangle_size = triangles.size() * sizeof(float4);
	int total_num_triangles = triangles.size() / 3;
	total_number_of_triangles = total_num_triangles;

	if (triangle_size > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, &triangles[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bindTriangles(dev_triangle_p, total_num_triangles, triangle_texture);
	}

	// compute scene bounding box by merging bounding boxes of individual meshes
	//与多个mesh比较得到bounding box
	scene_aabbox_min = mesh1.bounding_box[0];
	scene_aabbox_max = mesh1.bounding_box[1];
	std::cout << "obj bounding box: min:(" << scene_aabbox_min.x << "," << scene_aabbox_min.y << "," << scene_aabbox_min.z << ") max:"
		<< scene_aabbox_max.x << "," << scene_aabbox_max.y << "," << scene_aabbox_max.z << ")" << std::endl;

}




__device__ float3 lightReflect(float3 p, float3 n)
{
	float d = dot(p, n);
	return p - 2 * d*n;
}

__device__ bool lightRefract(float3 p, float3 n, float ni_over_nt,float3 &refracted)
{
	float3 uv = normalize(p);
	float dt = dot(uv, n);
	float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1 - dt*dt);
	if (discriminant > 0)
	{
		refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
		return true;
	}
	else
		return false;


}

// SCENE
// 9 spheres forming a Cornell box
// small enough to be in constant GPU memory
// { float radius, { float3 position }, { float3 emission }, { float3 colour }, refl_type }
__constant__ Sphere spheres[] = {
	{ 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f }, DIFF ,0.0f}, //Left 
	{ 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f }, DIFF,0.0f }, //Rght 
	{ 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f }, DIFF,0.0f }, //Back 
	{ 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f }, DIFF ,0.0f }, //Frnt 
	{ 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .25f }, DIFF ,0.0f }, //Botm 
	{ 1e5f, { 50.0f, -1e5f + 81.1f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f  }, DIFF ,0.0f }, //Top 
	//{ 16.5f, { 47.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, METAL ,0.0f }, // small sphere 1
	{ 10.5f, { 73.0f, 10.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f }, METAL ,0.0f }, // small sphere 2
	{ 8.5f,{ 27.0f, 8.5f, 120.0f },{ 0.0f, 0.0f, 0.0f },{ 1.0f, 1.0f, 1.0f }, GLASS ,2.5f }, // small sphere 2
	{ 600.0f, { 50.0f, 681.3f - .77f, 81.6f }, { 2.0f, 2.0f, 2.0f }, { 0.0f, 0.0f, 0.0f }, DIFF ,0.0f }  // Light
};

__constant__ Box boxes[] = {
	// FORMAT: { float3 minbounds,    float3 maxbounds,         float3 emission,    float3 colour,       Refl_t }
	{ { 55.0f, 0.1f, 113.0f },{ 85.0f, 27.0f, 115.0f },{ .0f, .0f, 0.0f },{ 1.0f, 1.0f, 1.0f }, GLASS,1.1f },
	{ { 16.6f, 0.1f, 11.5f },{ 63.2f, 18.93f, 47.6f },{ .0f, .0f, 0.0f },{ .25f, .75f, .75f }, DIFF,1.1f }
};

__device__ inline bool intersect_scene(const Ray &r, float &t, int &sphere_id, int &box_id, int& triangle_id, const int number_of_triangles,  int &geomtype, const float3& bbmin, const float3& bbmax) {

	float tmin = 1e20;
	float tmax = -1e20;
	float d = 1e21;
	float k = 1e21;
	float q = 1e21;
	float inf = t = 1e20;

	// SPHERES
	// intersect all spheres in the scene
	float numspheres = sizeof(spheres) / sizeof(Sphere);
	for (int i = int(numspheres); i--;)  // for all spheres in scene
										 // keep track of distance from origin to closest intersection point
		if ((d = spheres[i].intersect_sphere(r)) && d < t) 
		{ 
			t = d; sphere_id = i;
			geomtype = 1;
		}

	// BOXES
	// intersect all boxes in the scene
	float numboxes = sizeof(boxes) / sizeof(Box);
	for (int i = int(numboxes); i--;) // for all boxes in scene
		if ((k = boxes[i].intersect(r)) && k < t) 
		{ 
			t = k; box_id = i; 
			geomtype = 2; 
		}

	Box scene_bbox; // bounding box around triangle meshes
	scene_bbox.min = bbmin;
	scene_bbox.max = bbmax;
	int ji = 0;
	//intersectAllTriangles(r, t, triangle_id, number_of_triangles, geomtype);
	//// if ray hits bounding box of triangle meshes, intersect ray with all triangles
	//printf("111\n");
	if (scene_bbox.intersect(r)) {
		intersectAllTriangles(r, t, triangle_id, number_of_triangles, geomtype);
		//intersectAllTriangles(r, t, triangle_id, number_of_triangles, geomtype);
	}
	// t is distance to closest intersection of ray with all primitives in the scene (spheres, boxes and triangles)
	return t<inf;
}



// radiance function, the meat of path tracing 
// solves the rendering equation: 
// outgoing radiance (at a point) = emitted radiance + reflected radiance
// reflected radiance is sum (integral) of incoming radiance from all directions in hemisphere above point, 
// multiplied by reflectance function of material (BRDF) and cosine incident angle 
__device__ float3 radiance(Ray &r, unsigned int *s1, unsigned int *s2, curandState *randstate, const int totaltris, const float3& scene_aabb_min, const float3& scene_aabb_max)
{
	float factor = 1.9;
	float3 accucolor = make_float3(0.0f, 0.0f, 0.0f); // accumulates ray colour with each iteration through bounce loop
	float3 mask = make_float3(1.0f, 1.0f, 1.0f); 

	// ray bounce loop (no Russian Roulette used) 
	for (int bounces = 0; bounces < 7; bounces++)
	{  // iteration up to 4 bounces (replaces recursion in CPU code)

		float t;           // distance to closest intersection 
		int sphere_id = -1;
		int box_id = -1;   // index of intersected sphere
		int triangle_id = -1;
		int geomtype = -1;
		Refl_t refltype;
		float3 f;  // primitive colour
		float3 emit; // primitive emission colour
		float3 x; // intersection point 
		float3 n; // normal
		float3 nl; // oriented normal
		float3 d; // ray direction of next path segment
		float refaction_info;
	// test ray for intersection with scene
		if (!intersect_scene(r, t, sphere_id, box_id, triangle_id, totaltris, geomtype, scene_aabb_min, scene_aabb_max))
			return make_float3(0.0f, 0.0f, 0.0f); // if miss, return black
		//printf("%d\n", geomtype);
		//std::cout << geomtype << std::endl;
		if (geomtype == 1)
		{
			// else, we've got a hit!
			// compute hitpoint and normal
			const Sphere &obj = spheres[sphere_id];  // hitobject
			x = r.orig + r.dir*t;          // hitpoint 
			n = normalize(x - obj.pos);    // normal
			nl = dot(n, r.dir) < 0 ? n : n * -1; // front facing normal
			f = obj.col;   // object colour
			refaction_info = obj.refraction;
			refltype = obj.refl;
			emit = obj.emi;  // object emission
			accucolor += (mask * emit);
		}
		if (geomtype == 2) 
		{
			Box &box = boxes[box_id];
			x = r.orig + r.dir*t;  // intersection point on object
			n = normalize(box.normalAt(x)); // normal
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal
			f = box.col;  // box colour
			refltype = box.refl;
			emit = box.emi; // box emission
			refaction_info = box.refraction;
			accucolor += (mask * emit);
		}
		// if triangle:
		if (geomtype == 3) {
			int tri_index = triangle_id;
			x = r.orig + r.dir*t;  // intersection point
			n = normalize(getTriangleNormal(tri_index));  // normal 
			nl = dot(n, r.dir) < 0 ? n : n * -1;  // correctly oriented normal

												  // colour, refltype and emit value are hardcoded and apply to all triangles
												  // no per triangle material support yet
			f = make_float3(1.0f, 1.0f, 1.0f);  // triangle colour
			refltype = DIFF;
			emit = make_float3(0.0f, 0.0f, 0.0f);
			accucolor += (mask * emit);
		}
		// all spheres in the scene are diffuse
		// diffuse material reflects light uniformly in all directions
		// generate new diffuse ray:
		// origin = hitpoint of previous ray in path
		// random direction in hemisphere above hitpoint (see "Realistic Ray Tracing", P. Shirley)
		if (refltype == DIFF)
		{
			// create 2 random numbers
			float r1 = 2 * M_PI *  getrandom(s1, s2); // pick random number on unit circle (radius = 1, circumference = 2*Pi) for azimuth
			float r2 = getrandom(s1, s2);  // pick random number for elevation
			float r2s = sqrtf(r2);
			// printf("%f\n", r2);
			// compute local orthonormal basis uvw at hitpoint to use for calculation random ray direction 
			// first vector = normal at hitpoint, second vector is orthogonal to first, third vector is orthogonal to first two vectors
			float3 w = nl;
			float3 u = normalize(cross((fabs(w.x) > .1 ? make_float3(0, 1, 0) : make_float3(1, 0, 0)), w));
			float3 v = cross(w, u);

			// compute random ray direction on hemisphere using polar coordinates
			// cosine weighted importance sampling (favours ray directions closer to normal direction)
			float3 d = normalize(u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrtf(1 - r2));

			// new ray origin is intersection point of previous ray with scene
			r.orig = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
			r.dir = d;

			mask *= f;    // multiply with colour of object       
			mask *= dot(d, nl);  // weigh light contribution using cosine of angle between incident light and normal
			mask *= factor;          // fudge factor
		}
		if (refltype == METAL)
		{
			r.orig = x + nl*0.05f; // offset ray origin slightly to prevent self intersection
			float3 incidient = r.dir;
			r.dir = lightReflect(incidient, nl);
			mask *= f;
			mask *= dot(r.dir, nl);  // weigh light contribution using cosine of angle between incident light and normal
			//mask *= 2;
		}
		if (refltype == GLASS)
		{
			float ni_over_nt;
			float3 reflected = lightReflect(r.dir, nl);
			float3 attenuation = make_float3(1.0, 1.0, 0.0);
			float3 refracted;
			if (dot(r.dir, n) > 0) ni_over_nt = refaction_info;
			else  ni_over_nt = 1 / refaction_info;
			if (lightRefract(r.dir, nl, ni_over_nt, refracted))
			{
				r.dir = refracted;
				r.orig = x - nl*0.05f;
				mask *= f;
				//mask *= dot(r.dir, nl);
				mask *= factor;
				//mask *= dot(r.dir, nl);
			}
			else
			{
				// printf("%f  %f  %f\n", refracted.x, refracted.y, refracted.z);
				r.dir = reflected;
				r.orig = x + nl*0.05f;
				mask *= f;
				mask *= dot(r.dir, nl);
				mask *= factor;
				//mask *= dot(r.dir, nl);
			}
		}

	}
	return accucolor;
}


// __global__ : executed on the device (GPU) and callable only from host (CPU) 
// this kernel runs in parallel on all the CUDA threads

__global__ void render_kernel(float3 *output, float3 *sumrecord, int* samples, int framenumber, uint hashedframenumber ,const int numtriangles, float3 scene_bbmin, float3 scene_bbmax) {

	// assign a CUDA thread to every pixel (x,y) 
	// blockIdx, blockDim and threadIdx are CUDA specific keywords
	// replaces nested outer loops in CPU code looping over image rows and image columns 
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;   
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	if (x >= width || y >= height)return;
	unsigned int i = (height - y - 1)*width + x; // index of current pixel (calculated using thread index) 
	
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);
	int size = 100;
	unsigned int s1 = x + size*curand_uniform(&randState);  // seeds for random number generator
	unsigned int s2 = y + size*curand_uniform(&randState);
// generate ray directed at lower left corner of the screen
// compute directions for all other rays by adding cx and cy increments in x and y direction
	Ray cam(make_float3(50, 52, 295.6), normalize(make_float3(0, -0.042612, -1))); // first hardcoded camera ray(origin, direction) 
	float3 cx = make_float3(width * .5135 / height, 0.0f, 0.0f); // ray direction offset in x direction
	float3 cy = normalize(cross(cx, cam.dir)) * .5135; // ray direction offset in y direction (.5135 is field of view angle)
	float3 r; // r is final pixel color       
    
	r = make_float3(0.0f); // reset r to zero for every pixel 

	for (int s = 0; s < samps; s++){  // samples per pixel
		
	// compute primary ray direction
		//printf("%f\n", getrandom(&s1, &s2));
		float3 d = cam.dir + cx*((curand_uniform(&randState) - 0.5 + x) / width - .5) + cy*((curand_uniform(&randState) - 0.5 + y) / height - .5);
		//float3 d = cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5) + cam.dir;
	//float3 d = cam.dir + cx*((.25 + x) / width - .5) + cy*((.25 + y) / height - .5);
	
	// create primary ray, add incoming radiance to pixelcolor
		r = r + radiance(Ray(cam.orig + d * 40, normalize(d)), &s1, &s2, &randState, numtriangles, scene_bbmin, scene_bbmax);
	}       // Camera rays are pushed ^^^^^ forward to start in interior 
	sumrecord[i].x += r.x;
	sumrecord[i].y += r.y;
	sumrecord[i].z += r.z;
	r.x = sumrecord[i].x / samples[i];
	r.y = sumrecord[i].y / samples[i];
	r.z = sumrecord[i].z / samples[i];
	//printf("%d\n", samples[i]);
	//cudaDeviceSynchronize();
	// write rgb value of pixel to image buffer on the GPU, clamp value to [0.0f, 1.0f] range
	output[i] = make_float3(clamp(r.x, 0.0f, 1.0f), clamp(r.y, 0.0f, 1.0f), clamp(r.z, 0.0f, 1.0f));
}

inline float clamp(float x){ return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; } 

inline int toInt(float x){ return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }  // convert RGB float in range [0,1] to int in range [0, 255] and perform gamma correction

int main(){
	initCUDAmemoryTriMesh();

	cv::Mat record = cv::Mat::zeros(height, width, CV_8UC3);
	float3* output_h = new float3[width*height]; // pointer to memory for image on the host (system RAM)
	float3* sumrecord_h = new float3[width*height];
	int* samples_h = new int[width*height];
	for (int i = 0; i < width*height; i++)
	{
		sumrecord_h[i].x = 0;
		sumrecord_h[i].y = 0;
		sumrecord_h[i].z = 0;
		samples_h[0];
	}
	float3* output_d;    // pointer to memory for image on the device (GPU VRAM)
	float3* sumrecord_d;
	int* samples_d;
	
	//for (int k = 0; k < 10; k++)
	//while(1)
	for (int k = 0; k < 1000; k++)
	{
		//cudaThreadSynchronize();
		// allocate memory on the CUDA device (GPU VRAM)
		cudaMalloc(&output_d, width * height * sizeof(float3));
		cudaMalloc(&sumrecord_d, width * height * sizeof(float3));
		cudaMalloc(&samples_d, width * height * sizeof(int));
		// dim3 is CUDA specific type, block and grid are required to schedule CUDA threads over streaming multiprocessors
		dim3 block(32, 32, 1);
		dim3 grid(width / block.x, height / block.y, 1);
		cudaMemcpy(sumrecord_d, sumrecord_h, width * height * sizeof(float3), cudaMemcpyHostToDevice);
		for (int i = 0; i < width*height; i++)
		{
			samples_h[i] += samps;
		}
		cudaMemcpy(samples_d, samples_h, width * height * sizeof(int), cudaMemcpyHostToDevice);
		//printf("CUDA initialised.\nStart rendering...\n");

		// schedule threads on device and launch CUDA kernel from host
		render_kernel << < grid, block >> > (output_d, sumrecord_d, samples_d, frames, WangHash(frames), total_number_of_triangles, scene_aabbox_max, scene_aabbox_min);


		// copy results of computation from device back to host
		cudaMemcpy(output_h, output_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
		cudaMemcpy(sumrecord_h, sumrecord_d, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
		// free CUDA memory
		cudaFree(output_d);
		cudaFree(sumrecord_d);
		cudaFree(samples_d);
		//printf("Done!\n");

		

		for (int index = 0; index < height*width; index++)
		{
			int i, j;
			i = index  % width;
			j = index / width;//存在倒置
							  //printf("c:%d, r:%d  %f, %f, %f\n\n", i, j, pixel[index].x, pi xel[index].y, pixel[index].z);
			record.at<cv::Vec3b>(j, i)(0) = toInt(output_h[index].z);
			record.at<cv::Vec3b>(j, i)(1) = toInt(output_h[index].y);
			record.at<cv::Vec3b>(j, i)(2) = toInt(output_h[index].x);
			
		}

		frames++;
		//std::cout << sumrecord_h[ji].z<<"  "<< sumrecord_h[ji].y<< "  "<< sumrecord_h[ji].x << std::endl;
		//cv::imshow("test", record);
		//cv::waitKey(3);
		//std::cout << "1" << std::endl;
	}
	
	cv::imwrite("1000_.jpg", record);
}
