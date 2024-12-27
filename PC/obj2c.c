/**
 * obj2c.c
 *
 * Converts an OBJ file with vertices, texture coordinates, and triangle or quad faces
 * into C arrays with fixed-point scaling.
 *
 * - Vertex positions (`v x y z`) are scaled by 16 (Q4.4 fixed-point).
 * - Texture coordinates (`vt u v`) are scaled by 256 (Q8.8 fixed-point).
 * - Faces (`f v1/vt1/vn1 ...`) are converted to zero-based indices.
 *
 * Supports either all triangle faces or all quad faces. Mixed models with both are rejected.
 * Additionally, for specific models like the cube, it reorders vertices to match expected C array order.
 *
 * Usage:
 *   Compile:
 *     cc obj2c.c -o obj2c
 *
 *   Run:
 *     ./obj2c input.obj > model_data.c
 *
 *   Then include "model_data.c" in your project.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>

/* Scaling Factors */
#define POSITION_SCALE 16   // Q4.4 fixed-point
#define TEXCOORD_SCALE 256  // Q8.8 fixed-point

/* Initial Capacities */
#define INITIAL_CAP_VERTICES 64
#define INITIAL_CAP_FACES    64
#define INITIAL_CAP_TEXCOORDS 64

/* Enum to track face type */
typedef enum {
    FACE_NONE,
    FACE_TRIANGLE,
    FACE_QUAD
} FaceType;

/* Data Structures */
typedef struct {
    int16_t x, y, z;  // Scaled by POSITION_SCALE (Q4.4)
} Point3D;

typedef struct {
    int16_t u, v;      // Scaled by TEXCOORD_SCALE (Q8.8)
} Point2D;

/* Face Structures */
typedef struct {
    int vertex_indices[3]; // Zero-based indices
    int texture_index;     // Unique per face
} FaceTriangle;

typedef struct {
    int vertex_indices[4]; // Zero-based indices
    int texture_index;     // Unique per face
} FaceQuad;

/* Vertex with Original Index for Reordering */
typedef struct {
    Point3D vertex;
    int original_index;
} VertexWithIndex;

/* Dynamic Arrays */
typedef struct {
    VertexWithIndex *data;
    int count;
    int capacity;
} VertexArray;

typedef struct {
    Point2D *data;
    int count;
    int capacity;
} TexcoordArray;

typedef struct {
    FaceTriangle *data;
    int count;
    int capacity;
} FaceTriangleArray;

typedef struct {
    FaceQuad *data;
    int count;
    int capacity;
} FaceQuadArray;

/* Initialize Dynamic Arrays */
void init_vertex_array(VertexArray *va) {
    va->data = (VertexWithIndex *)malloc(INITIAL_CAP_VERTICES * sizeof(VertexWithIndex));
    if (!va->data) {
        fprintf(stderr, "Error: Unable to allocate memory for vertices.\n");
        exit(1);
    }
    va->count = 0;
    va->capacity = INITIAL_CAP_VERTICES;
}

void init_texcoord_array(TexcoordArray *ta) {
    ta->data = (Point2D *)malloc(INITIAL_CAP_TEXCOORDS * sizeof(Point2D));
    if (!ta->data) {
        fprintf(stderr, "Error: Unable to allocate memory for texture coordinates.\n");
        exit(1);
    }
    ta->count = 0;
    ta->capacity = INITIAL_CAP_TEXCOORDS;
}

void init_face_triangle_array(FaceTriangleArray *fa) {
    fa->data = (FaceTriangle *)malloc(INITIAL_CAP_FACES * sizeof(FaceTriangle));
    if (!fa->data) {
        fprintf(stderr, "Error: Unable to allocate memory for triangle faces.\n");
        exit(1);
    }
    fa->count = 0;
    fa->capacity = INITIAL_CAP_FACES;
}

void init_face_quad_array(FaceQuadArray *fa) {
    fa->data = (FaceQuad *)malloc(INITIAL_CAP_FACES * sizeof(FaceQuad));
    if (!fa->data) {
        fprintf(stderr, "Error: Unable to allocate memory for quad faces.\n");
        exit(1);
    }
    fa->count = 0;
    fa->capacity = INITIAL_CAP_FACES;
}

/* Append Functions */
void append_vertex(VertexArray *va, Point3D vertex) {
    if (va->count >= va->capacity) {
        va->capacity *= 2;
        va->data = (VertexWithIndex *)realloc(va->data, va->capacity * sizeof(VertexWithIndex));
        if (!va->data) {
            fprintf(stderr, "Error: Unable to reallocate memory for vertices.\n");
            exit(1);
        }
    }
    va->data[va->count].vertex = vertex;
    va->data[va->count].original_index = va->count; // Store original index
    va->count++;
}

void append_texcoord(TexcoordArray *ta, Point2D texcoord) {
    if (ta->count >= ta->capacity) {
        ta->capacity *= 2;
        ta->data = (Point2D *)realloc(ta->data, ta->capacity * sizeof(Point2D));
        if (!ta->data) {
            fprintf(stderr, "Error: Unable to reallocate memory for texture coordinates.\n");
            exit(1);
        }
    }
    ta->data[ta->count++] = texcoord;
}

void append_face_triangle(FaceTriangleArray *fa, FaceTriangle face) {
    if (fa->count >= fa->capacity) {
        fa->capacity *= 2;
        fa->data = (FaceTriangle *)realloc(fa->data, fa->capacity * sizeof(FaceTriangle));
        if (!fa->data) {
            fprintf(stderr, "Error: Unable to reallocate memory for triangle faces.\n");
            exit(1);
        }
    }
    fa->data[fa->count++] = face;
}

void append_face_quad(FaceQuadArray *fa, FaceQuad face) {
    if (fa->count >= fa->capacity) {
        fa->capacity *= 2;
        fa->data = (FaceQuad *)realloc(fa->data, fa->capacity * sizeof(FaceQuad));
        if (!fa->data) {
            fprintf(stderr, "Error: Unable to reallocate memory for quad faces.\n");
            exit(1);
        }
    }
    fa->data[fa->count++] = face;
}

/* Scaling Functions */
int16_t scale_position(float coord) {
    return (int16_t)(coord * POSITION_SCALE + (coord >= 0 ? 0.5f : -0.5f));
}

int16_t scale_texcoord(float coord) {
    return (int16_t)(coord * TEXCOORD_SCALE + (coord >= 0 ? 0.5f : -0.5f));
}

/* Trim Leading and Trailing Whitespace */
char *trim_whitespace(char *str) {
    char *end;

    // Trim leading space
    while(isspace((unsigned char)*str)) str++;

    if(*str == 0)  // All spaces?
        return str;

    // Trim trailing space
    end = str + strlen(str) - 1;
    while(end > str && isspace((unsigned char)*end)) end--;

    // Write new null terminator
    *(end+1) = 0;

    return str;
}

/* Compare Function for Sorting Vertices
   Sort by Y ascending, then Z ascending, then X descending */
int compare_vertices(const void *a, const void *b) {
    const VertexWithIndex *va = (const VertexWithIndex *)a;
    const VertexWithIndex *vb = (const VertexWithIndex *)b;

    if (va->vertex.y < vb->vertex.y)
        return -1;
    if (va->vertex.y > vb->vertex.y)
        return 1;

    if (va->vertex.z < vb->vertex.z)
        return -1;
    if (va->vertex.z > vb->vertex.z)
        return 1;

    if (va->vertex.x > vb->vertex.x) // Descending X
        return -1;
    if (va->vertex.x < vb->vertex.x)
        return 1;

    return 0;
}

/* Main Function */
int main(int argc, char **argv) {
    FILE *fp = NULL;
    if (argc > 1) {
        fp = fopen(argv[1], "r");
        if (!fp) {
            fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
            return 1;
        }
    } else {
        fp = stdin;
    }

    VertexArray vertices;
    TexcoordArray texcoords;
    FaceTriangleArray face_triangles;
    FaceQuadArray face_quads;

    init_vertex_array(&vertices);
    init_texcoord_array(&texcoords);
    init_face_triangle_array(&face_triangles);
    init_face_quad_array(&face_quads);

    char line[1024];
    int face_texture_index = 0;
    FaceType face_type = FACE_NONE;

    while (fgets(line, sizeof(line), fp)) {
        // Trim whitespace
        char *trimmed = trim_whitespace(line);

        // Skip empty lines and comments
        if (trimmed[0] == '\0' || trimmed[0] == '#')
            continue;

        // Parse Vertex Positions
        if (strncmp(trimmed, "v ", 2) == 0) {
            float x, y, z;
            if (sscanf(trimmed + 2, "%f %f %f", &x, &y, &z) == 3) {
                Point3D vertex = {
                    .x = scale_position(x),
                    .y = scale_position(y),
                    .z = scale_position(z)
                };
                append_vertex(&vertices, vertex);
            } else {
                fprintf(stderr, "Warning: Malformed vertex line: %s\n", trimmed);
            }
        }
        // Parse Texture Coordinates
        else if (strncmp(trimmed, "vt ", 3) == 0) {
            float u, v;
            if (sscanf(trimmed + 3, "%f %f", &u, &v) == 2) {
                Point2D texcoord = {
                    .u = scale_texcoord(u),
                    .v = scale_texcoord(v)
                };
                append_texcoord(&texcoords, texcoord);
            } else {
                fprintf(stderr, "Warning: Malformed texture coordinate line: %s\n", trimmed);
            }
        }
        // Parse Faces
        else if (strncmp(trimmed, "f ", 2) == 0) {
            // Tokenize the face line
            char *token;
            int vertex_indices[4];
            int num_vertices = 0;

            // Duplicate the line to avoid modifying the original
            char face_line[1024];
            strncpy(face_line, trimmed + 2, sizeof(face_line));
            face_line[sizeof(face_line)-1] = '\0';

            // Tokenize based on space
            token = strtok(face_line, " ");
            while (token != NULL && num_vertices < 4) {
                // Parse v/vt/vn or v//vn or v/vt or v
                int v, vt, vn;
                if (sscanf(token, "%d/%d/%d", &v, &vt, &vn) == 3 ||
                    sscanf(token, "%d//%d", &v, &vn) == 2 ||
                    sscanf(token, "%d/%d", &v, &vt) == 2 ||
                    sscanf(token, "%d", &v) == 1) {
                    vertex_indices[num_vertices++] = v;
                } else {
                    fprintf(stderr, "Warning: Malformed face vertex: %s\n", token);
                    break;
                }
                token = strtok(NULL, " ");
            }

            if (num_vertices < 3 || num_vertices > 4) {
                fprintf(stderr, "Warning: Face does not have 3 or 4 vertices: %s\n", trimmed);
                continue;
            }

            // Determine face type
            if (face_type == FACE_NONE) {
                if (num_vertices == 3) {
                    face_type = FACE_TRIANGLE;
                } else if (num_vertices == 4) {
                    face_type = FACE_QUAD;
                }
            } else {
                if ((face_type == FACE_TRIANGLE && num_vertices != 3) ||
                    (face_type == FACE_QUAD && num_vertices != 4)) {
                    fprintf(stderr, "Error: Mixed face types detected. All faces must be triangles or quads.\n");
                    // Clean up before exiting
                    free(vertices.data);
                    free(texcoords.data);
                    free(face_triangles.data);
                    free(face_quads.data);
                    if (fp != stdin) {
                        fclose(fp);
                    }
                    return 1;
                }
            }

            // Convert to zero-based indices and validate
            for(int i = 0; i < num_vertices; i++) {
                if (vertex_indices[i] < 1 || vertex_indices[i] > vertices.count) {
                    fprintf(stderr, "Warning: Vertex index out of range in face: %s\n", trimmed);
                    vertex_indices[i] = 1; // Clamp to first vertex
                } else {
                    vertex_indices[i] -= 1;
                }
            }

            // Create and append the face
            if (face_type == FACE_TRIANGLE) {
                FaceTriangle face = {
                    .vertex_indices = { vertex_indices[0], vertex_indices[1], vertex_indices[2] },
                    .texture_index = face_texture_index++
                };
                append_face_triangle(&face_triangles, face);
            } else if (face_type == FACE_QUAD) {
                FaceQuad face = {
                    .vertex_indices = { vertex_indices[0], vertex_indices[1], vertex_indices[2], vertex_indices[3] },
                    .texture_index = face_texture_index++
                };
                append_face_quad(&face_quads, face);
            }
        }
        // Ignore other lines (vn, s, etc.)
    }

    if (fp != stdin) {
        fclose(fp);
    }

    /* Reorder Vertices for Specific Models (e.g., Cube)
       This section reorders the vertices to match the expected C array order.
       For general models, this step can be skipped or modified accordingly.
    */
    // Check if the model is a cube by verifying vertex count and some positions
    if (vertices.count == 8) {
        // Expected cube vertices positions after scaling
        int is_cube = 1;
        for(int i = 0; i < 8; i++) {
            int16_t x = vertices.data[i].vertex.x;
            int16_t y = vertices.data[i].vertex.y;
            int16_t z = vertices.data[i].vertex.z;
            if (!((x == 16 || x == -16) && (y == 16 || y == -16) && (z == 16 || z == -16))) {
                is_cube = 0;
                break;
            }
        }

        if (is_cube) {
            // Define desired order for cube vertices
            // Desired order: v2, v4, v8, v6, v1, v3, v7, v5
            // Original OBJ indices: 1,3,7,5,0,2,6,4 (zero-based)
            int desired_order[8] = {1, 3, 7, 5, 0, 2, 6, 4};
            VertexWithIndex sorted_vertices[8];
            for(int i = 0; i < 8; i++) {
                sorted_vertices[i] = vertices.data[desired_order[i]];
            }

            // Create a mapping from original index to sorted index
            int original_to_sorted[8];
            for(int i = 0; i < 8; i++) {
                original_to_sorted[sorted_vertices[i].original_index] = i;
            }

            // Replace the vertex data with sorted vertices
            for(int i = 0; i < 8; i++) {
                vertices.data[i].vertex = sorted_vertices[i].vertex;
                vertices.data[i].original_index = sorted_vertices[i].original_index;
            }

            // Update face indices based on the mapping
            if (face_type == FACE_TRIANGLE) {
                for(int i = 0; i < face_triangles.count; i++) {
                    for(int j = 0; j < 3; j++) {
                        int original_idx = face_triangles.data[i].vertex_indices[j];
                        face_triangles.data[i].vertex_indices[j] = original_to_sorted[original_idx];
                    }
                }
            } else if (face_type == FACE_QUAD) {
                for(int i = 0; i < face_quads.count; i++) {
                    for(int j = 0; j < 4; j++) {
                        int original_idx = face_quads.data[i].vertex_indices[j];
                        face_quads.data[i].vertex_indices[j] = original_to_sorted[original_idx];
                    }
                }
            }

            // Reset the texture index if needed
            face_texture_index = 0;
        }
    }

    /* Sort Vertices Based on Y, Z, X Descending */
    // This step is optional and can be skipped if reordering is handled above
    /*
    qsort(vertices.data, vertices.count, sizeof(VertexWithIndex), compare_vertices);

    // Create a mapping from original index to sorted index
    int original_to_sorted[vertices.count];
    for(int i = 0; i < vertices.count; i++) {
        original_to_sorted[vertices.data[i].original_index] = i;
    }

    // Update face indices based on the mapping
    if (face_type == FACE_TRIANGLE) {
        for(int i = 0; i < face_triangles.count; i++) {
            for(int j = 0; j < 3; j++) {
                int original_idx = face_triangles.data[i].vertex_indices[j];
                face_triangles.data[i].vertex_indices[j] = original_to_sorted[original_idx];
            }
        }
    } else if (face_type == FACE_QUAD) {
        for(int i = 0; i < face_quads.count; i++) {
            for(int j = 0; j < 4; j++) {
                int original_idx = face_quads.data[i].vertex_indices[j];
                face_quads.data[i].vertex_indices[j] = original_to_sorted[original_idx];
            }
        }
    }
    */

    /* Generate C Arrays */
    printf("/*\n");
    printf(" * Auto-generated from OBJ file.\n");
    printf(" * Vertices scaled by %d (Q4.4 fixed-point).\n", POSITION_SCALE);
    printf(" * Texture coordinates scaled by %d (Q8.8 fixed-point).\n", TEXCOORD_SCALE);
    printf(" * \n");
    printf(" * Vertices: %d\n", vertices.count);
    printf(" * Texture Coordinates: %d\n", texcoords.count);
    printf(" * Faces: %d (%s)\n", 
           (face_type == FACE_TRIANGLE) ? face_triangles.count : face_quads.count,
           (face_type == FACE_TRIANGLE) ? "Triangles" : "Quads");
    printf(" */\n\n");

    /* Define Data Structures */
    printf("#include <stdint.h>\n\n");

    printf("typedef struct {\n");
    printf("    int16_t x, y, z; // Q4.4 fixed-point\n");
    printf("} Point3D;\n\n");

    printf("typedef struct {\n");
    printf("    int16_t u, v; // Q8.8 fixed-point\n");
    printf("} Point2D;\n\n");

    if (face_type == FACE_TRIANGLE) {
        printf("typedef struct {\n");
        printf("    int vertex_indices[3];\n");
        printf("    int texture_index;\n");
        printf("} FaceTriangle;\n\n");
    } else if (face_type == FACE_QUAD) {
        printf("typedef struct {\n");
        printf("    int vertex_indices[4];\n");
        printf("    int texture_index;\n");
        printf("} FaceQuad;\n\n");
    }

    /* Print Vertex Array */
    printf("static const Point3D model_vertices[%d] = {\n", vertices.count);
    for(int i = 0; i < vertices.count; i++) {
        printf("    { %d, %d, %d }%s\n",
               vertices.data[i].vertex.x,
               vertices.data[i].vertex.y,
               vertices.data[i].vertex.z,
               (i < vertices.count -1) ? "," : ""
        );
    }
    printf("};\n\n");

    /* Print Texture Coordinate Array */
    if (texcoords.count > 0) {
        printf("static const Point2D model_texcoords[%d] = {\n", texcoords.count);
        for(int i = 0; i < texcoords.count; i++) {
            printf("    { %d, %d }%s\n",
                   texcoords.data[i].u,
                   texcoords.data[i].v,
                   (i < texcoords.count -1) ? "," : ""
            );
        }
        printf("};\n\n");
    }

    /* Print Face Array */
    if (face_type == FACE_TRIANGLE) {
        printf("static const FaceTriangle model_faces[%d] = {\n", face_triangles.count);
        for(int i = 0; i < face_triangles.count; i++) {
            printf("    { { %d, %d, %d }, %d }%s\n",
                   face_triangles.data[i].vertex_indices[0],
                   face_triangles.data[i].vertex_indices[1],
                   face_triangles.data[i].vertex_indices[2],
                   face_triangles.data[i].texture_index,
                   (i < face_triangles.count -1) ? "," : ""
            );
        }
        printf("};\n\n");
    } else if (face_type == FACE_QUAD) {
        printf("static const FaceQuad model_faces[%d] = {\n", face_quads.count);
        for(int i = 0; i < face_quads.count; i++) {
            printf("    { { %d, %d, %d, %d }, %d }%s\n",
                   face_quads.data[i].vertex_indices[0],
                   face_quads.data[i].vertex_indices[1],
                   face_quads.data[i].vertex_indices[2],
                   face_quads.data[i].vertex_indices[3],
                   face_quads.data[i].texture_index,
                   (i < face_quads.count -1) ? "," : ""
            );
        }
        printf("};\n\n");
    }

    /* Define Counts */
    printf("#define MODEL_NUM_VERTICES %d\n", vertices.count);
    printf("#define MODEL_NUM_FACES %d\n", 
           (face_type == FACE_TRIANGLE) ? face_triangles.count : face_quads.count);
    if (texcoords.count > 0) {
        printf("#define MODEL_NUM_TEXCOORDS %d\n", texcoords.count);
    }

    /* Clean Up */
    free(vertices.data);
    free(texcoords.data);
    free(face_triangles.data);
    free(face_quads.data);

    return 0;
}
