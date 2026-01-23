pub struct Icosahedron {
    pub vertices: [[f32; 3]; 12],
    pub triangles: [[u16; 3]; 20],
}

impl Icosahedron {
    pub fn new(inner_radius: f32) -> Self {
        // http://blog.andreaskahler.com/2009/06/creating-icosphere-mesh-in-code.html
        let t0 = (1.0 + 5.0f32.sqrt()) / 2.0;
        let s0 = 1.0;
        let r = (t0 + s0) / 3.0;
        let scale = inner_radius / r;
        let t = t0 * scale;
        let s = 1.0 * scale;
        Self {
            vertices: [
                [-s, t, 0.0],
                [s, t, 0.0],
                [-s, -t, 0.0],
                [s, -t, 0.0],
                [0.0, -s, t],
                [0.0, s, t],
                [0.0, -s, -t],
                [0.0, s, -t],
                [t, 0.0, -s],
                [t, 0.0, s],
                [-t, 0.0, -s],
                [-t, 0.0, s],
            ],
            triangles: [
                // 5 faces around point 0
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                // 5 adjacent faces
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                // 5 faces around point 3
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                // 5 adjacent faces
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
        }
    }
}
