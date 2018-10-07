#include"Object.h"


// read triangle data from obj file
void loadObj(const std::string filename, TriangleMesh &mesh)
{
	std::ifstream in(filename.c_str());

	if (!in.good())
	{
		std::cout << "ERROR: loading obj:(" << filename << ") file not found or not good" << "\n";
		system("PAUSE");
		exit(0);
	}

	char buffer[256], str[255];
	float f1, f2, f3;

	while (!in.getline(buffer, 255).eof())
	{
		buffer[255] = '\0';
		sscanf_s(buffer, "%s", str, 255);

		// reading a vertex
		if (buffer[0] == 'v' && (buffer[1] == ' ' || buffer[1] == 32)) {
			if (sscanf(buffer, "v %f %f %f", &f1, &f2, &f3) == 3) {
				mesh.verts.push_back(make_float3(f1, f2, f3));
			}
			else {
				std::cout << "ERROR: vertex not in wanted format in OBJLoader" << "\n";
				exit(-1);
			}
		}

		// reading faceMtls 
		else if (buffer[0] == 'f' && (buffer[1] == ' ' || buffer[1] == 32))
		{
			TriangleFace f;
			int nt = sscanf(buffer, "f %d %d %d", &f.v[0], &f.v[1], &f.v[2]);
			if (nt != 3) {
				std::cout << "ERROR: I don't know the format of that FaceMtl" << "\n";
				exit(-1);
			}

			mesh.faces.push_back(f);
		}
	}

	// calculate the bounding box of the mesh
	mesh.bounding_box[0] = make_float3(1000000, 1000000, 1000000);
	mesh.bounding_box[1] = make_float3(-1000000, -1000000, -1000000);
	for (unsigned int i = 0; i < mesh.verts.size(); i++)
	{
		//update min and max value
		mesh.bounding_box[0] = fminf(mesh.verts[i], mesh.bounding_box[0]);
		mesh.bounding_box[1] = fmaxf(mesh.verts[i], mesh.bounding_box[1]);
	}

	std::cout << "obj file loaded: number of faces:" << mesh.faces.size() << " number of vertices:" << mesh.verts.size() << std::endl;
	std::cout << "obj bounding box: min:(" << mesh.bounding_box[0].x << "," << mesh.bounding_box[0].y << "," << mesh.bounding_box[0].z << ") max:"
		<< mesh.bounding_box[1].x << "," << mesh.bounding_box[1].y << "," << mesh.bounding_box[1].z << ")" << std::endl;
}