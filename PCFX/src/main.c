#include "pcfx.h"
#include "bg.h"
#include "textures.h"
#include "title.h"
#include "trig.h"

extern int nframe;

#include "gamepal.h"

#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#define KRAM_PAGE0 0x00000000
#define KRAM_PAGE1 0x80000000
#define ADPCM_OFFSET 0x00000000 | KRAM_PAGE1

int OFFSET_PAGE; 

const int32_t kram_pages[2] = { KRAM_PAGE1, KRAM_PAGE0};

// Separate assembly function for writing to KRAM
static inline void write_kram(int16_t value) {
    asm volatile (
        "out.h %0, 0x604[r0]\n\t"
        :
        : "r"(value)
        :
    );
}



#define MAX_VERTICES 1024*20
#define MAX_FACES 1024*20

int16_t framebuffer_game[256*240/2];

#define GAME_FRAMEBUFFER framebuffer_game
#define BIG_ENDIAN 1 
#define _16BITS_WRITES 1


static inline void my_memcpy32(void *dest, const void *src, size_t n) 
{
    int32_t *d = (int32_t*)dest;
    const int32_t *s = (const int32_t*)src;

    // Copy 4 bytes at a time (32 bits)
    while (n >= 4) {
        *d++ = *s++;
        n -= 4;
    }

    // Handle any remaining bytes (if the size is not a multiple of 4)
    int8_t *d8 = (int8_t*)d;
    const int8_t *s8 = (const int8_t*)s;
    while (n > 0) {
        *d8++ = *s8++;
        n--;
    }
}

static char* myitoa(int value) {
    static char buffer[12];  // Enough for an integer (-2147483648 has 11 characters + 1 for '\0')
    char* ptr = buffer + sizeof(buffer) - 1;
    int is_negative = 0;

    // Null-terminate the buffer
    *ptr = '\0';

    // Handle negative numbers
    if (value < 0) {
        is_negative = 1;
        value = -value;
    }

    // Process each digit
    do {
        *--ptr = (value % 10) + '0';
        value /= 10;
    } while (value);

    // Add the negative sign if necessary
    if (is_negative) {
        *--ptr = '-';
    }

    return ptr;
}

// ─────────────────────────────────────────────────────────────────────────────
// If you have hardware division, define HARDWARE_DIV. Otherwise, we use a LUT.
// ─────────────────────────────────────────────────────────────────────────────
// #define HARDWARE_DIV

// ─────────────────────────────────────────────────────────────────────────────
// Division LUT Setup
// ─────────────────────────────────────────────────────────────────────────────
#define DIV_TAB_SIZE   1024
#define DIV_TAB_HALF   (DIV_TAB_SIZE / 2)
#define DIV_TAB_SHIFT  16

static int32_t divTab[DIV_TAB_SIZE];

void initDivs(void)
{
    for (int i = 0; i < DIV_TAB_SIZE; i++) {
        int ii = i - DIV_TAB_HALF; // range: -512..+511
        if (ii == 0) {
            ii++;  // avoid /0
        }
        // store reciprocal in 16.16 fixed
        divTab[i] = (1 << DIV_TAB_SHIFT) / ii;
    }
}

#ifdef HARDWARE_DIV
    #define Division(numerator, denominator) ( (numerator) / (denominator) )
#else
    // LUT-based division => (num * reciprocal(den)) >> 16
    // Make sure denominator + 512 is within [0..1023]
    #define Division(numerator, denominator) \
        ( ((numerator) * divTab[(denominator) + DIV_TAB_HALF]) >> DIV_TAB_SHIFT )
#endif


// ─────────────────────────────────────────────────────────────────────────────
// Compile-time switch for using 32×32→32 multiplication (with 64-bit intermediate).
// Comment it out if you want 16×16→32.
// ─────────────────────────────────────────────────────────────────────────────
//#define USE_32X32_64

// ─────────────────────────────────────────────────────────────────────────────
// Configurable fixed integer type
// If using 32×32 multiply, we want the inputs to be 32-bit. Otherwise 16-bit.
// ─────────────────────────────────────────────────────────────────────────────
#ifdef USE_32X32_64
typedef int32_t fixint;  // the “basic” type we’ll use for transforms
#else
typedef int16_t fixint;  // default to 16-bit
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Multiplication macros/functions
//  - For 16×16→32, we do normal 32-bit multiply.
//  - For 32×32→(64)32, we do a 64-bit intermediate
// ─────────────────────────────────────────────────────────────────────────────


#define FIXED_POINT_SHIFT  8
#define FIXED_POINT_SCALE  (1 << FIXED_POINT_SHIFT)

#ifdef USE_32X32_64
static inline int32_t MUL(fixint a, fixint b) {
    // 64-bit intermediate, returns 32-bit
    int32_t r = (int32_t)a * (int32_t)b;
    return r;
}
#else
static inline int32_t MUL(fixint a, fixint b) {
    // 16×16 => 32
    return ((int16_t)a * (int16_t)b);
}
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Basic definitions
// ─────────────────────────────────────────────────────────────────────────────
#define BASE_SCREEN_WIDTH  256
#define BASE_SCREEN_HEIGHT 240
#define SCREEN_WIDTH       256
#define SCREEN_HEIGHT      240
#define SCREEN_WIDTH_HALF  (SCREEN_WIDTH / 2)
#define SCREEN_HEIGHT_HALF (SCREEN_HEIGHT / 2)

// Projection distance (changed slightly for example)
#define BASE_PROJECTION_DISTANCE -160
#define PROJECTION_DISTANCE ((int16_t)( ( (int32_t)BASE_PROJECTION_DISTANCE * SCREEN_HEIGHT) / BASE_SCREEN_HEIGHT ))

#define STARTING_Z_OFFSET  -256

// Angle settings
#define ANGLE_MAX  256
#define ANGLE_MASK (ANGLE_MAX - 1)


// ─────────────────────────────────────────────────────────────────────────────
// 3D point, 2D point structures
// ─────────────────────────────────────────────────────────────────────────────
typedef struct {
    fixint x, y, z;
} Point3D;

typedef struct {
    int16_t x, y;   // screen coords
    int16_t u, v;   // fixed-point texture coords (8.8)
} Point2D;

// Face definition: 4 vertex indices + a texture index
typedef struct {
    fixint vertex_indices[4];
    int16_t texture_index;
} Face;

typedef struct {
    fixint vertex_indices[3];  // Indices of the three vertices that form the triangle
    int16_t texture_index;      // Index of the texture to apply to this triangle
} Face_Tris;

// “Rendered” face
typedef struct {
    Point2D projected_vertices[4];
    int32_t average_depth;
    int16_t     texture_index;
} FaceToDraw;

// Global face list, face_count
FaceToDraw face_list[MAX_FACES];
int        face_count = 0;

typedef struct {
    Point2D projected_vertices[3]; // Projected 2D vertices of the triangle
    int32_t average_depth;         // Average depth for sorting
    int16_t     texture_index;         // Texture index for rendering
} FaceToDrawTri;

// Global list and counter for triangle faces
FaceToDrawTri tri_face_list[MAX_FACES];
int tri_face_count = 0;
// ─────────────────────────────────────────────────────────────────────────────
// Global EdgeData for drawTexturedQuad
// ─────────────────────────────────────────────────────────────────────────────
typedef struct {
    int16_t y_start, y_end;
    int32_t x, x_step;
    int16_t u, v;
    int16_t u_step, v_step;
} EdgeData;

static EdgeData edges[4];  // Global instead of local

// ─────────────────────────────────────────────────────────────────────────────
// swap_elements: static inline, used by qsort_game
// ─────────────────────────────────────────────────────────────────────────────
static inline void swap_elements(void* aa, void* bb, size_t sz) {
    uint8_t *pa = (uint8_t*)aa;
    uint8_t *pb = (uint8_t*)bb;
    while (sz--) {
        uint8_t tmp = *pa;
        *pa++ = *pb;
        *pb++ = tmp;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compare_faces: same as before
// ─────────────────────────────────────────────────────────────────────────────
static inline int compare_faces(const void *a, const void *b) {
    const FaceToDraw *fa = (const FaceToDraw *)a;
    const FaceToDraw *fb = (const FaceToDraw *)b;
    return (fa->average_depth - fb->average_depth);
}

// ─────────────────────────────────────────────────────────────────────────────
// qsort_game using pointer arithmetic & our swap_elements
// ─────────────────────────────────────────────────────────────────────────────
static void qsort_game(void *base, size_t n, size_t size,
                       int (*compar)(const void*, const void*)) 
{
    if (n < 2) return;

    typedef struct { int start, end; } StackItem;
    StackItem stack[64];
    int       top = 0;

    stack[top++] = (StackItem){0, (int)n - 1};

    while (top > 0) {
        StackItem range = stack[--top];
        int start = range.start;
        int end   = range.end;
        if (start >= end) continue;

        uint8_t *pivot = (uint8_t*)base + (end * size);

        int i = start - 1;
        for (int j = start; j < end; j++) {
            uint8_t *elem_j = (uint8_t*)base + (j * size);
            if (compar(elem_j, pivot) <= 0) {
                i++;
                swap_elements((uint8_t*)base + (i * size), elem_j, size);
            }
        }
        i++;
        swap_elements((uint8_t*)base + (i * size), pivot, size);

        int p = i;
        if (p - 1 > start) {
            stack[top++] = (StackItem){start, p - 1};
        }
        if (p + 1 < end) {
            stack[top++] = (StackItem){p + 1, end};
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// initialize_trigonometry
// ─────────────────────────────────────────────────────────────────────────────
void initialize_trigonometry() {
    /*for (int angle = 0; angle < ANGLE_MAX; angle++) {
        double radians  = angle * 2.0 * M_PI / ANGLE_MAX;
        double sineVal  = sin(radians);
        double cosVal   = cos(radians);

        sin_lookup[angle] = (fixint)(FIXED_POINT_SCALE * sineVal);
        cos_lookup[angle] = (fixint)(FIXED_POINT_SCALE * cosVal);
    }*/
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotations
// ─────────────────────────────────────────────────────────────────────────────
static inline Point3D rotateX(Point3D p, fixint angle) {
    fixint sinA = sin_lookup[angle & ANGLE_MASK];
    fixint cosA = cos_lookup[angle & ANGLE_MASK];

    int32_t y = (MUL(p.y, cosA) - MUL(p.z, sinA)) >> FIXED_POINT_SHIFT;
    int32_t z = (MUL(p.y, sinA) + MUL(p.z, cosA)) >> FIXED_POINT_SHIFT;
    return (Point3D){ p.x, (fixint)y, (fixint)z };
}

static inline Point3D rotateY(Point3D p, fixint angle) {
    fixint sinA = sin_lookup[angle & ANGLE_MASK];
    fixint cosA = cos_lookup[angle & ANGLE_MASK];

    int32_t x = (MUL(p.x, cosA) + MUL(p.z, sinA)) >> FIXED_POINT_SHIFT;
    int32_t z = (MUL(p.z, cosA) - MUL(p.x, sinA)) >> FIXED_POINT_SHIFT;
    return (Point3D){ (fixint)x, p.y, (fixint)z };
}

static inline Point3D rotateZ(Point3D p, fixint angle) {
    fixint sinA = sin_lookup[angle & ANGLE_MASK];
    fixint cosA = cos_lookup[angle & ANGLE_MASK];

    int32_t x = (MUL(p.x, cosA) - MUL(p.y, sinA)) >> FIXED_POINT_SHIFT;
    int32_t y = (MUL(p.x, sinA) + MUL(p.y, cosA)) >> FIXED_POINT_SHIFT;
    return (Point3D){ (fixint)x, (fixint)y, p.z };
}

// ─────────────────────────────────────────────────────────────────────────────
// Project a 3D point (now using Division macro instead of / denom)
// ─────────────────────────────────────────────────────────────────────────────
static inline Point2D project(Point3D p, int16_t u, int16_t v) {
    int16_t distance = PROJECTION_DISTANCE;
    fixint  denom    = (fixint)(distance - p.z);
    
    // Not required
	//if (denom == 0) denom = 1;  // avoid /0

    // factor = (distance << FIXED_POINT_SHIFT) / denom
    // replaced with: factor = Division( (distance << SHIFT), denom )
    int32_t numerator = ((int32_t)distance << FIXED_POINT_SHIFT);
    int32_t factor    = Division(numerator, denom);

    int32_t x = (MUL(p.x, (fixint)factor)) >> FIXED_POINT_SHIFT;
    int32_t y = (MUL(p.y, (fixint)factor)) >> FIXED_POINT_SHIFT;

    return (Point2D){ (int16_t)x, (int16_t)y, u, v };
}



#include <stdint.h>


// ─────────────────────────────────────────────────────────────────────────────
// Constants and External Variables
// ─────────────────────────────────────────────────────────────────────────────
// Global framebuffer pointer
static int16_t *fb_ptr;

// ─────────────────────────────────────────────────────────────────────────────
// SetPixel8_xs and SetPixel8_xe for Both Endiannesses
// ─────────────────────────────────────────────────────────────────────────────

#define BIGENDIAN_SYSTEM 1

static inline int16_t GetPixel16() {
    //return *fb_ptr;
    return eris_king_kram_read();
}

static inline void SetPixel8_xs(int16_t color) {
	int16_t pixel = GetPixel16();
#ifdef BIGENDIAN_SYSTEM
    // In big endian, the first byte is the higher byte
    write_kram((pixel & 0xFF00) | ((int16_t)(color) << 8));
#elif defined(LITTLEENDIAN_SYSTEM)
    // In little endian, the first byte is the lower byte
    write_kram((pixel & 0x00FF) | ((int16_t)(color) << 8));
#endif
}

static inline void SetPixel8_xe(int16_t color) {
	uint16_t pixel = GetPixel16();
#ifdef BIGENDIAN_SYSTEM
    // In big endian, the second byte is the lower byte
	write_kram((pixel & 0x00FF) | (int16_t)(color));
#elif defined(LITTLEENDIAN_SYSTEM)
    // In little endian, the first byte is the lower byte
    write_kram((pixel & 0xFF00) | (int16_t)(color));
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// SetPixel16
// ─────────────────────────────────────────────────────────────────────────────

static inline void SetPixel16(int16_t colors) {
    // Set two pixels at once
    //*fb_ptr = colors;
    write_kram(colors);
}

// ─────────────────────────────────────────────────────────────────────────────
// fetchTextureColor
// ─────────────────────────────────────────────────────────────────────────────

static inline int32_t fetchTextureColor(int32_t u, int32_t v, int texture_index) {
    int16_t tex_x = (int16_t)((u >> 8) & 31);
    int16_t tex_y = (int16_t)((v >> 8) & 31);

    int16_t block_y_offset = (int16_t)(texture_index * 32);
    int16_t final_y = block_y_offset + tex_y;

    return texture[final_y * 32 + tex_x];
}

// ─────────────────────────────────────────────────────────────────────────────
// fetchTextureColor16
// ─────────────────────────────────────────────────────────────────────────────

static inline int32_t fetchTextureColor16(int32_t u, int32_t v, int32_t du, int32_t dv, int texture_index) {
    // First pixel coordinates
    int16_t tex_x1 = (int16_t)((u >> 8) & 31);
    int16_t tex_y1 = (int16_t)((v >> 8) & 31);

    // Second pixel coordinates
    int32_t u2 = u + du;
    int32_t v2 = v + dv;
    int16_t tex_x2 = (int16_t)((u2 >> 8) & 31);
    int16_t tex_y2 = (int16_t)((v2 >> 8) & 31);

    // Calculate texture indices
    int16_t block_y_offset = (int16_t)(texture_index * 32);
    int16_t final_y1 = block_y_offset + tex_y1;
    int16_t final_y2 = block_y_offset + tex_y2;

    int index1 = final_y1 * 16 + tex_x1;
    int index2 = final_y2 * 16 + tex_x2;

    // Fetch colors
    int16_t color1 = texture[index1];


    return color1;
}

// ─────────────────────────────────────────────────────────────────────────────
// drawScanline
// ─────────────────────────────────────────────────────────────────────────────

static inline void drawScanline(int32_t xs, int32_t xe,
                                int32_t u,  int32_t v,
                                int32_t du, int32_t dv,
                                int32_t y,  int texture_index)
{
    // Initialize framebuffer pointer to the starting position
    //fb_ptr = (uint16_t *)GAME_FRAMEBUFFER + y * (SCREEN_WIDTH / 2) + (xs / 2);
	eris_king_set_kram_write(y * (SCREEN_WIDTH / 2) + (xs / 2) + (OFFSET_PAGE), 1);
	asm volatile (
			"movea %0, r0, r10 \n\t"
			"out.h r10, 0x600[r0] \n\t"
			: 
			: "i"(0xE)
			: "r10", "r0"
	);

    // Handle the first pixel if xs is odd
   /* if (xs & 1) {
        int16_t color = fetchTextureColor(u, v, texture_index);
        SetPixel8_xs(color);
        xs++;
        u += du;
        v += dv;
        //fb_ptr++; // Move to the next 16-bit word
    }*/

    // Calculate the number of remaining pixels
    int32_t remaining_pixels = xe - xs + 1;
    int32_t num_pairs = remaining_pixels / 2;

    // Process pixel pairs
    while (num_pairs) 
    {
		int16_t colors = fetchTextureColor16(u, v, du, dv, texture_index);
        u += du;
        v += dv;

        
        u += du;
        v += dv;

        // Set the combined 16-bit color to the framebuffer
        SetPixel16(colors);

        // Advance the framebuffer pointer
        //fb_ptr++;
        num_pairs--;
    }

    // Handle the last pixel if the remaining number of pixels is odd
    /*if (remaining_pixels & 1) {
		int16_t color = fetchTextureColor(u, v, texture_index);
        SetPixel8_xe(color);
    }*/
}


static inline void drawTexturedTriangle(Point2D p1, Point2D p2, Point2D p3, int texture_index) {
    // Sort the vertices by y-coordinate ascending (p1.y <= p2.y <= p3.y)
    if (p1.y > p2.y) { Point2D temp = p1; p1 = p2; p2 = temp; }
    if (p1.y > p3.y) { Point2D temp = p1; p1 = p3; p3 = temp; }
    if (p2.y > p3.y) { Point2D temp = p2; p2 = p3; p3 = temp; }

    int32_t total_height = p3.y - p1.y;
    if (total_height == 0) return; // Avoid division by zero

    // Calculate the inverse slope and texture step between p1 and p3
    int32_t dx13 = Division( (p3.x - p1.x) << FIXED_POINT_SHIFT, total_height );
    int32_t du13 = Division( (p3.u - p1.u), total_height );
    int32_t dv13 = Division( (p3.v - p1.v), total_height );

    // Initialize variables for the two segments of the triangle
    int32_t dx12 = 0, du12 = 0, dv12 = 0;
    int32_t dx23 = 0, du23 = 0, dv23 = 0;

    int32_t segment_height = p2.y - p1.y;
    if (segment_height > 0) {
        dx12 = Division( (p2.x - p1.x) << FIXED_POINT_SHIFT, segment_height );
        du12 = Division( (p2.u - p1.u), segment_height );
        dv12 = Division( (p2.v - p1.v), segment_height );
    }

    if (p3.y - p2.y > 0) {
        int32_t segment_dy = p3.y - p2.y;
        dx23 = Division( (p3.x - p2.x) << FIXED_POINT_SHIFT, segment_dy );
        du23 = Division( (p3.u - p2.u), segment_dy );
        dv23 = Division( (p3.v - p2.v), segment_dy );
    }

    // Starting points
    int32_t x_start = p1.x << FIXED_POINT_SHIFT;
    int32_t u_start = p1.u;
    int32_t v_start = p1.v;

    int32_t x_end = p1.x << FIXED_POINT_SHIFT;
    int32_t u_end = p1.u;
    int32_t v_end = p1.v;

    int32_t y;

    // First half of the triangle
    for (y = p1.y; y < p2.y; y++) {
        int32_t xs = x_start >> FIXED_POINT_SHIFT;
        int32_t xe = x_end >> FIXED_POINT_SHIFT;

        int32_t us = u_start;
        int32_t vs = v_start;

        int32_t ue = u_end;
        int32_t ve = v_end;

        // Ensure xs <= xe
        if (xs > xe) {
            int32_t temp_x = xs; xs = xe; xe = temp_x;
            int32_t temp_u = us; us = ue; ue = temp_u;
            int32_t temp_v = vs; vs = ve; ve = temp_v;
        }

        int32_t dx = xe - xs;
        int32_t du = 0, dv = 0;
        if (dx > 0) {
            du = Division( (ue - us), dx );
            dv = Division( (ve - vs), dx );
        }

        // Draw the horizontal scanline for this y
        drawScanline(xs, xe, us, vs, du, dv, y, texture_index);

        // Increment the starting and ending points
        x_start += dx13;
        u_start += du13;
        v_start += dv13;

        if (y < p2.y) {
            x_end += dx12;
            u_end += du12;
            v_end += dv12;
        }
    }

    // Second half of the triangle
    x_end = p2.x << FIXED_POINT_SHIFT;
    u_end = p2.u;
    v_end = p2.v;

    for (; y <= p3.y; y++) {
        int32_t xs = x_start >> FIXED_POINT_SHIFT;
        int32_t xe = x_end >> FIXED_POINT_SHIFT;

        int32_t us = u_start;
        int32_t vs = v_start;

        int32_t ue = u_end;
        int32_t ve = v_end;

        // Ensure xs <= xe
        if (xs > xe) {
            int32_t temp_x = xs; xs = xe; xe = temp_x;
            int32_t temp_u = us; us = ue; ue = temp_u;
            int32_t temp_v = vs; vs = ve; ve = temp_v;
        }

        int32_t dx = xe - xs;
        int32_t du = 0, dv = 0;
        if (dx > 0) {
            du = Division( (ue - us), dx );
            dv = Division( (ve - vs), dx );
        }

        // Draw the horizontal scanline for this y
        drawScanline(xs, xe, us, vs, du, dv, y, texture_index);

        // Increment the starting and ending points
        x_start += dx13;
        u_start += du13;
        v_start += dv13;

        x_end += dx23;
        u_end += du23;
        v_end += dv23;
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// drawTexturedQuad
// We replaced divisions in edge building with Division() macro
// ─────────────────────────────────────────────────────────────────────────────
static inline void drawTexturedQuad(Point2D p0, Point2D p1, Point2D p2, Point2D p3,
                                    int texture_index)
{
    Point2D points[4] = { p0, p1, p2, p3 };
    Point2D *pp = points;

    // Unrolled min_y, max_y
    fixint min_y = pp->y;
    fixint max_y = pp->y;

    if ((pp + 1)->y < min_y) min_y = (pp + 1)->y; else if ((pp + 1)->y > max_y) max_y = (pp + 1)->y;
    if ((pp + 2)->y < min_y) min_y = (pp + 2)->y; else if ((pp + 2)->y > max_y) max_y = (pp + 2)->y;
    if ((pp + 3)->y < min_y) min_y = (pp + 3)->y; else if ((pp + 3)->y > max_y) max_y = (pp + 3)->y;

    EdgeData *edp     = edges;
    EdgeData *edp_end = edges + 4;

    // Build edges
    for (int i = 0; edp < edp_end; edp++, i++)
    {
        Point2D *pA = pp + i;
        Point2D *pB = pp + ((i + 1) & 3);

        int16_t dy = (int16_t)(pB->y - pA->y);
        if (dy == 0) {
            edp->y_start = pA->y;
            edp->y_end   = pA->y;
            edp->x       = (int32_t)pA->x << FIXED_POINT_SHIFT;
            edp->x_step  = 0;
            edp->u       = pA->u;
            edp->v       = pA->v;
            edp->u_step  = 0;
            edp->v_step  = 0;
            continue;
        }

        if (dy > 0) {
            edp->y_start = pA->y;
            edp->y_end   = pB->y;
            edp->x       = (int32_t)pA->x << FIXED_POINT_SHIFT;
            edp->u       = pA->u;
            edp->v       = pA->v;

            // x_step = ((pB->x - pA->x) << FIXED_POINT_SHIFT) / dy
            int32_t dx_num = ((int32_t)(pB->x - pA->x) << FIXED_POINT_SHIFT);
            edp->x_step = Division(dx_num, dy);

            // u_step = (pB->u - pA->u) / dy
            int32_t du_num = (pB->u - pA->u);
            edp->u_step = (int16_t)Division(du_num, dy);

            // v_step = ...
            int32_t dv_num = (pB->v - pA->v);
            edp->v_step = (int16_t)Division(dv_num, dy);
        }
        else {
            edp->y_start = pB->y;
            edp->y_end   = pA->y;
            edp->x       = (int32_t)pB->x << FIXED_POINT_SHIFT;
            edp->u       = pB->u;
            edp->v       = pB->v;

            dy = (int16_t)(-dy);

            int32_t dx_num = ((int32_t)(pA->x - pB->x) << FIXED_POINT_SHIFT);
            edp->x_step = Division(dx_num, dy);

            int32_t du_num = (pA->u - pB->u);
            edp->u_step = (int16_t)Division(du_num, dy);

            int32_t dv_num = (pA->v - pB->v);
            edp->v_step = (int16_t)Division(dv_num, dy);
        }
    }

    // Render scanlines
    for (int16_t y = min_y; y <= max_y; y++)
    {
        int16_t num_intersections = 0;
        int16_t x_int[4], u_int[4], v_int[4];

        EdgeData *scan = edges;
        for (int i = 0; i < 4; i++, scan++)
        {
            if (y >= scan->y_start && y < scan->y_end)
            {
                x_int[num_intersections] = (int16_t)(scan->x >> FIXED_POINT_SHIFT);
                u_int[num_intersections] = scan->u;
                v_int[num_intersections] = scan->v;
                num_intersections++;

                scan->x += scan->x_step;
                scan->u += scan->u_step;
                scan->v += scan->v_step;
            }
        }

        if (num_intersections < 2) continue;

        if (x_int[0] > x_int[1]) {
            int16_t tmpx = x_int[0]; x_int[0] = x_int[1]; x_int[1] = tmpx;
            int16_t tmpu = u_int[0]; u_int[0] = u_int[1]; u_int[1] = tmpu;
            int16_t tmpv = v_int[0]; v_int[0] = v_int[1]; v_int[1] = tmpv;
        }

        int16_t xs = x_int[0];
        int16_t xe = x_int[1];
        int16_t us = u_int[0];
        int16_t vs = v_int[0];
        int16_t ue = u_int[1];
        int16_t ve = v_int[1];

        int16_t dx = (int16_t)(xe - xs);
        if (dx == 0) continue;

        // du, dv also replaced with Division
        int16_t du = (int16_t)Division((ue - us), dx);
        int16_t dv = (int16_t)Division((ve - vs), dx);

        drawScanline(xs, xe, us, vs, du, dv, y, texture_index);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// draw_3d_object
// ─────────────────────────────────────────────────────────────────────────────
void draw_3d_object_quads(
    Point3D *base_vertices, int num_vertices,
    Face    *faces,         int num_faces,
    fixint angle_x, fixint angle_y, fixint angle_z,
    fixint translate_x, fixint translate_y, fixint translate_z)
{
    if (num_faces > MAX_FACES) return;

    static Point3D transformed[1024];
    if (num_vertices > 1024) return;

    // Transform
    Point3D *src  = base_vertices;
    Point3D *dest = transformed;
    for (int i = 0; i < num_vertices; i++) {
        Point3D v = *src++;
        v = rotateX(v, angle_x);
        v = rotateY(v, angle_y);
        v = rotateZ(v, angle_z);
        v.x += translate_x;
        v.y += translate_y;
        v.z += translate_z;
        *dest++ = v;
    }

    face_count = 0;
    Face *fptr = faces;
    for (int i = 0; i < num_faces; i++, fptr++) {
        fixint idx0 = (fixint)fptr->vertex_indices[0];
        fixint idx1 = (fixint)fptr->vertex_indices[1];
        fixint idx2 = (fixint)fptr->vertex_indices[2];
        fixint idx3 = (fixint)fptr->vertex_indices[3];
        if (idx0 < 0 || idx0 >= num_vertices) continue;
        if (idx1 < 0 || idx1 >= num_vertices) continue;
        if (idx2 < 0 || idx2 >= num_vertices) continue;
        if (idx3 < 0 || idx3 >= num_vertices) continue;

        Point3D *tv0 = &transformed[idx0];
        Point3D *tv1 = &transformed[idx1];
        Point3D *tv2 = &transformed[idx2];
        Point3D *tv3 = &transformed[idx3];

        fixint ax = (fixint)(tv1->x - tv0->x);
        fixint ay = (fixint)(tv1->y - tv0->y);
        fixint az = (fixint)(tv1->z - tv0->z);

        fixint bx = (fixint)(tv2->x - tv0->x);
        fixint by = (fixint)(tv2->y - tv0->y);
        fixint bz = (fixint)(tv2->z - tv0->z);

        int32_t nx = MUL(ay, bz) - MUL(az, by);
        int32_t ny = MUL(az, bx) - MUL(ax, bz);
        int32_t nz = MUL(ax, by) - MUL(ay, bx);

        // back-face culling if nz < 0
        if (nz < 0) 
        {
			//printf("Culled\n");
			continue;
		}
		
        // Project
        Point2D p0 = project(*tv0, 0,        0);
        Point2D p1 = project(*tv1, (32<<8),  0);
        Point2D p2 = project(*tv2, (32<<8), (32<<8));
        Point2D p3 = project(*tv3, 0,       (32<<8));

        p0.x += SCREEN_WIDTH_HALF;  p0.y += SCREEN_HEIGHT_HALF;
        p1.x += SCREEN_WIDTH_HALF;  p1.y += SCREEN_HEIGHT_HALF;
        p2.x += SCREEN_WIDTH_HALF;  p2.y += SCREEN_HEIGHT_HALF;
        p3.x += SCREEN_WIDTH_HALF;  p3.y += SCREEN_HEIGHT_HALF;

        int32_t z_sum = (tv0->z + tv1->z + tv2->z + tv3->z);

        FaceToDraw *fd = &face_list[face_count++];
        fd->projected_vertices[0] = p0;
        fd->projected_vertices[1] = p1;
        fd->projected_vertices[2] = p2;
        fd->projected_vertices[3] = p3;
        fd->average_depth         = z_sum;
        fd->texture_index         = fptr->texture_index;
    }

    // Sort
    qsort_game(face_list, face_count, sizeof(FaceToDraw), compare_faces);

    // Draw
    FaceToDraw *fl = face_list;
    for (int i = 0; i < face_count; i++, fl++) {
        drawTexturedQuad(
            fl->projected_vertices[0],
            fl->projected_vertices[1],
            fl->projected_vertices[2],
            fl->projected_vertices[3],
            fl->texture_index
        );
    }
}


Point3D cube_vertices[8] = {
    { 16, -16, -16 },
    { 16, -16, 16 },
    { -16, -16, 16 },
    { -16, -16, -16 },
    { 16, 16, -16 },
    { 16, 16, 16 },
    { -16, 16, 16 },
    { -16, 16, -16 }
};

Face cube_faces[6] = {
    { { 4, 7, 6, 5 }, 0 },
    { { 1, 5, 6, 2 }, 1 },
    { { 2, 6, 7, 3 }, 2 },
    { { 3, 0, 1, 2 }, 3 },
    { { 0, 4, 5, 1 }, 4 },
    { { 3, 7, 4, 0 }, 5 }
};



// For each line v X Y Z => finalX = round(X*16)
static const Point3D tank_vertices[40] =
{
    { -10, 3, 22 },
    { -10, 14, 22 },
    { -10, 3, -21 },
    { -10, 14, -21 },
    { 10, 3, 22 },
    { 10, 14, 22 },
    { 10, 3, -21 },
    { 10, 14, -21 },
    { -7, 14, 19 },
    { -7, 23, 19 },
    { -7, 14, 5 },
    { -7, 23, 5 },
    { 7, 14, 19 },
    { 7, 23, 19 },
    { 7, 14, 5 },
    { 7, 23, 5 },
    { -2, 17, 38 },
    { -2, 20, 38 },
    { -2, 17, 19 },
    { -2, 20, 19 },
    { 2, 17, 38 },
    { 2, 20, 38 },
    { 2, 17, 19 },
    { 2, 20, 19 },
    { 10, 0, 20 },
    { 10, 13, 20 },
    { 10, 0, -20 },
    { 10, 13, -20 },
    { 13, 0, 20 },
    { 13, 13, 20 },
    { 13, 0, -20 },
    { 13, 13, -20 },
    { -10, 0, 20 },
    { -10, 13, 20 },
    { -10, 0, -20 },
    { -10, 13, -20 },
    { -13, 0, 20 },
    { -13, 13, 20 },
    { -13, 0, -20 },
    { -13, 13, -20 }
};


static const Face tank_faces[30] =
{
    { { 0,1,3,2 }, 0 },
    { { 2,3,7,6 }, 1 },
    { { 6,7,5,4 }, 2 },
    { { 4,5,1,0 }, 3 },
    { { 2,6,4,0 }, 4 },
    { { 7,3,1,5 }, 5 },

    { { 8,9,11,10 }, 0 },
    { { 10,11,15,14 }, 1 },
    { { 14,15,13,12 }, 2 },
    { { 12,13,9,8 }, 3 },
    { { 10,14,12,8 }, 4 },
    { { 15,11,9,13 }, 5 },


    { { 16,17,19,18 }, 0 },
    { { 18,19,23,22 }, 1 },
    { { 22,23,21,20 }, 2 },
    { { 20,21,17,16 }, 3 },
    { { 18,22,20,16 }, 4 },
    { { 23,19,17,21 }, 5 },

    { { 24,25,27,26 }, 0 },
    { { 26,27,31,30 }, 1 },
    { { 30,31,29,28 }, 2 },
    { { 28,29,25,24 }, 3 },
    { { 26,30,28,24 }, 4 },
    { { 31,27,25,29 }, 5 },

    { { 32,34,35,33 }, 0 },
    { { 34,38,39,35 }, 1 },
    { { 38,36,37,39 }, 2 },
    { { 36,32,33,37 }, 3 },
    { { 34,32,36,38 }, 4 },
    { { 39,37,33,35 }, 5 },
};


    // Sort the triangle faces from farthest to nearest based on average_depth
    // Define a comparison function for qsort_game
static inline int compare_faces_tri(const void *a, const void *b) 
{
	const FaceToDrawTri *fa = (const FaceToDrawTri *)a;
	const FaceToDrawTri *fb = (const FaceToDrawTri *)b;
	// Sort in descending order (farthest first)
	return (fb->average_depth - fa->average_depth);
}

void draw_3d_object_tris(
    Point3D *base_vertices, int num_vertices,
    Face_Tris    *faces,         int num_faces,
    fixint angle_x, fixint angle_y, fixint angle_z,
    fixint translate_x, fixint translate_y, fixint translate_z)
{
    if (num_faces > MAX_FACES) return;

    static Point3D transformed[MAX_VERTICES];
    if (num_vertices > MAX_VERTICES) return;

    // Transform all vertices
    Point3D *src  = base_vertices;
    Point3D *dest = transformed;
    for (int i = 0; i < num_vertices; i++) {
        Point3D v = *src++;
        v = rotateX(v, angle_x);
        v = rotateY(v, angle_y);
        v = rotateZ(v, angle_z);
        v.x += translate_x;
        v.y += translate_y;
        v.z += translate_z;
        *dest++ = v;
    }

    // Reset triangle face count
    tri_face_count = 0;

    // Process each triangle face
    Face_Tris *fptr = faces;
    for (int i = 0; i < num_faces; i++, fptr++) {
        int idx0 = fptr->vertex_indices[0];
        int idx1 = fptr->vertex_indices[1];
        int idx2 = fptr->vertex_indices[2];

        // Validate vertex indices
        if (idx0 < 0 || idx0 >= num_vertices) continue;
        if (idx1 < 0 || idx1 >= num_vertices) continue;
        if (idx2 < 0 || idx2 >= num_vertices) continue;

        Point3D *tv0 = &transformed[idx0];
        Point3D *tv1 = &transformed[idx1];
        Point3D *tv2 = &transformed[idx2];

        // Compute the normal vector for back-face culling
        fixint ax = tv1->x - tv0->x;
        fixint ay = tv1->y - tv0->y;
        fixint az = tv1->z - tv0->z;

        fixint bx = tv2->x - tv0->x;
        fixint by = tv2->y - tv0->y;
        fixint bz = tv2->z - tv0->z;

        int32_t nx = MUL(ay, bz) - MUL(az, by);
        int32_t ny = MUL(az, bx) - MUL(ax, bz);
        int32_t nz = MUL(ax, by) - MUL(ay, bx);

        // Back-face culling: skip triangles facing away
        if (nz < 0) continue;

        // Project the 3D vertices to 2D screen space
        // Assign texture coordinates based on your texture atlas layout
        // Example: Equilateral mapping within the texture block
        Point2D p0 = project(*tv0, 0,        0);
        Point2D p1 = project(*tv1, (32<<8),  0);
        Point2D p2 = project(*tv2, (16<<8), (32<<8));

        // Adjust to screen coordinates
        p0.x += SCREEN_WIDTH_HALF;  p0.y += SCREEN_HEIGHT_HALF;
        p1.x += SCREEN_WIDTH_HALF;  p1.y += SCREEN_HEIGHT_HALF;
        p2.x += SCREEN_WIDTH_HALF;  p2.y += SCREEN_HEIGHT_HALF;

        // Compute average depth for sorting (use average of z-values)
        int32_t z_sum = (tv0->z + tv1->z + tv2->z);
        int32_t avg_depth = z_sum / 3;

        // Store the projected triangle in the face list
        if (tri_face_count >= MAX_FACES) continue;

        FaceToDrawTri *fd = &tri_face_list[tri_face_count++];
        fd->projected_vertices[0] = p0;
        fd->projected_vertices[1] = p1;
        fd->projected_vertices[2] = p2;
        fd->average_depth         = avg_depth;
        fd->texture_index         = fptr->texture_index;
    }

    // Perform the sort
    qsort_game(tri_face_list, tri_face_count, sizeof(FaceToDrawTri), compare_faces_tri);

    // Render the sorted triangles
    FaceToDrawTri *fl = tri_face_list;
    for (int i = 0; i < tri_face_count; i++, fl++) {
        drawTexturedTriangle(
            fl->projected_vertices[0],
            fl->projected_vertices[1],
            fl->projected_vertices[2],
            fl->texture_index
        );
    }
}


    // Example triangle object (pyramid)
Point3D pyramid_vertices[5] = {
        {  0,  16,  0 },
        { 16, -16, 16 },
        { -16, -16, 16 },
        { -16, -16, -16 },
        { 16, -16, -16 }
};
Face_Tris pyramid_faces[6] = {
        { {0, 1, 2}, 0 },
        { {0, 2, 3}, 1 },
        { {0, 3, 4}, 2 },
        { {0, 4, 1}, 3 },
        { {1, 4, 3}, 4 },
        { {1, 3, 2}, 5 }
 };

static const Point3D model_vertices[8] = {
    { 16, -16, 16 },
    { -16, -16, -16 },
    { -16, 16, -16 },
    { 16, 16, 16 },
    { 16, -16, -16 },
    { -16, -16, 16 },
    { -16, 16, 16 },
    { 16, 16, -16 }
};

static const Face_Tris model_faces[12] = {
    { { 4, 0, 5 }, 0 },
    { { 4, 5, 1 }, 0 },
    { { 7, 2, 6 }, 1 },
    { { 7, 6, 3 }, 1 },
    { { 4, 7, 3 }, 2 },
    { { 4, 3, 0 }, 2 },
    { { 0, 3, 6 }, 3 },
    { { 0, 6, 5 }, 3 },
    { { 5, 6, 2 }, 4 },
    { { 5, 2, 1 }, 4 },
    { { 7, 4, 1 }, 5 },
    { { 7, 1, 2 }, 5 }
};

int32_t offset_page_flip_king[2] = {0, ((256*240)/2)/1024};
int32_t offset_page_flip[2] = {(256*240)/2, 0 };
int pageflip = 0;

__attribute__((optimize("unroll-loops")))
static inline void KingClear()
{
	eris_king_set_kram_write(0  /*| kram_pages[pageflip]*/ + (OFFSET_PAGE), 1);
    asm volatile (
        "movea %0, r0, r10 \n\t"
        "out.h r10, 0x600[r0] \n\t"
        : 
        : "i"(0xE)
        : "r10", "r0"
    );
    for (int i = 0; i < 3840; ++i) {
        write_kram(0);
        write_kram(0);
        write_kram(0);
        write_kram(0);
        write_kram(0);
        write_kram(0);
        write_kram(0);
        write_kram(0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
	

	eris_king_init();
	eris_tetsu_init();
	eris_pad_init(0);
	
	Empty_Palette();
	Clear_VDC(0);
	
	
	eris_low_cdda_set_volume(63,63);
	
	Set_Video(KING_BGMODE_256_PAL);
	Upload_Palette(gamepal);
	initTimer(0, 1423);
	
	Clear_VDC(0);

    // Initialize the division lookup table
    initDivs();

    bool running = true;
    fixint angle_x = 0;
    fixint angle_y = 0;
    fixint angle_z = 0;

	//my_memcpy32(GAME_FRAMEBUFFER, bg_game, SCREEN_WIDTH * SCREEN_HEIGHT);
	//eris_king_set_kram_write(0, 1);
	//king_kram_write_buffer(bg_game, SCREEN_WIDTH * SCREEN_HEIGHT); 
	
	pageflip = 0;
	OFFSET_PAGE =  (256*240)/2;
	
    while (running) {

        // Clear with background
		//eris_king_set_kram_write((OFFSET_PAGE), 1);
		//king_kram_write_buffer(bg_game, SCREEN_WIDTH * SCREEN_HEIGHT); 
		
		KingClear();

		//angle_x = (fixint)((angle_x + 1) & ANGLE_MASK);
        angle_y = (fixint)((angle_y + 2) & ANGLE_MASK);
        //angle_z = (fixint)((angle_z + 3) & ANGLE_MASK);

		// Draw 3D Cube
		/*draw_3d_object_quads(
            cube_vertices, 8,
            cube_faces,    6,
            angle_x, angle_y, angle_z,
            0, 0, STARTING_Z_OFFSET
        );*/
        
		draw_3d_object_quads(
            tank_vertices, 40,
            tank_faces,    30,
            angle_x, angle_y, angle_z,
            0, -20, STARTING_Z_OFFSET
        );
        
       // Draw 3D Pyramid (Triangles)
       //draw_3d_object_tris(
        //    pyramid_vertices, 5,
		//	pyramid_faces,   6,
        //    angle_x, angle_y, angle_z,
        //    0, 0, STARTING_Z_OFFSET
       // );*


		//eris_king_set_kram_write(0, 1);
		//king_kram_write_buffer(framebuffer_game, SCREEN_WIDTH * SCREEN_HEIGHT); */
        
		eris_king_set_bat_cg_addr(KING_BG0, 0, offset_page_flip_king[pageflip]);
		OFFSET_PAGE = offset_page_flip[pageflip];
		
		pageflip ^= 1;
        
		print_at(0, 1, 12, myitoa(getFps()));
		vsync(0);
		++nframe;
    }


    return 0;
}
