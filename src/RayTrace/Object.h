#pragma once
/*
	Objects in the simulation,include the function of the object data struct and
	the intersection algorithm.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "device_launch_parameters.h"
#include "cutil_math.h" // from http://www.icmc.usp.br/~castelo/CUDA/common/inc/cutil_math.h
#include <curand.h>
#include <curand_kernel.h>
#include <sstream>
#include <fstream>
#include <vector>

inline __device__ float3 minf3(float3 a, float3 b) { return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
inline __device__ float3 maxf3(float3 a, float3 b) { return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
inline __device__ float minf1(float a, float b) { return a < b ? a : b; }
inline __device__ float maxf1(float a, float b) { return a > b ? a : b; }

struct Ray
{
	float3 orig; // ray origin
	float3 dir;  // ray direction 
	__device__ Ray(float3 o_, float3 d_) : orig(o_), dir(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR, METAL, GLASS }; // material types, used in radiance(), only DIFF used here

struct Sphere
{
	float rad;            // radius 
	float3 pos, emi, col; // position, emission, colour 
	Refl_t refl;          // reflection type (e.g. diffuse)
	float refraction;     //index of refraction
	__device__ float intersect_sphere(const Ray &r) const
	{
		float3 op = pos - r.orig;    // distance from ray.orig to center sphere 
		float t, epsilon = 0.01f;  // epsilon required to prevent floating point precision artefacts
		float b = dot(op, r.dir);    // b in quadratic equation
		float disc = b*b - dot(op, op) + rad*rad;  // discriminant quadratic equation
		if (disc<0) return 0;       // if disc < 0, no real solution (we're not interested in complex roots) 
		else disc = sqrtf(disc);    // if disc >= 0, check for solutions using negative and positive discriminant
		return (t = b - disc)>epsilon ? t : ((t = b + disc)>epsilon ? t : 0); // pick closest point in front of ray origin
	}
};

struct Box {

	float3 min; // minimum bounds
	float3 max; // maximum bounds
	float3 emi; // emission
	float3 col; // colour
	Refl_t refl; // material type
	float refraction;     //index of refraction
	__device__ float intersect(const Ray &r) const
	{
		float epsilon = 0.001f; // required to prevent self intersection
		float3 tmin = (min - r.orig) / r.dir;
		float3 tmax = (max - r.orig) / r.dir;
		float3 real_min = minf3(tmin, tmax);
		float3 real_max = maxf3(tmin, tmax);
		float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
		float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);
		if (minmax >= maxmin) { return maxmin > epsilon ? maxmin : 0; }
		else return 0;
	}
	__device__ float3 Box::normalAt(float3 &point)
	{
		float3 normal = make_float3(0.f, 0.f, 0.f);
		float min_distance = 1e8;
		float distance;
		float epsilon = 0.001f;

		if (fabs(min.x - point.x) < epsilon) normal = make_float3(-1, 0, 0);
		else if (fabs(max.x - point.x) < epsilon) normal = make_float3(1, 0, 0);
		else if (fabs(min.y - point.y) < epsilon) normal = make_float3(0, -1, 0);
		else if (fabs(max.y - point.y) < epsilon) normal = make_float3(0, 1, 0);
		else if (fabs(min.z - point.z) < epsilon) normal = make_float3(0, 0, -1);
		else normal = make_float3(0, 0, 1);
		return normal;
	}
};

// helpers to load triangle data
struct TriangleFace
{
	int v[3]; // vertex indices
};
struct TriangleMesh
{
	std::vector<float3> verts;
	std::vector<TriangleFace> faces;
	float3 bounding_box[2];
};

void loadObj(const std::string filename, TriangleMesh &mesh); // forward declaration

															  // 1. load triangle mesh data from obj files
															  // 2. copy data to CPU memory (into vector<float4> triangles)
															  // 3. copy to CUDA global memory (allocated with dev_triangle_p pointer)
															  // 4. copy to CUDA texture memory with bindtriangles()
