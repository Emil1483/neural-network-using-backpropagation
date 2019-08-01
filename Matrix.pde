float e = 2.71828182846;

class Matrix {
  float[][] pos;
  int w;
  int h;
  
  Matrix(int h, int w) {
    this.h = h;
    this.w = w;
    pos = new float[h][w];
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        pos[i][j] = 1;
      }
    }
  }
  
  Matrix(int h) {
    this(h, 1);
  }
  
  Matrix(Matrix m) {
    h = m.pos.length;
    w = m.pos[0].length;
    pos = new float[h][w];
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        pos[i][j] = m.pos[i][j];
      }
    }
  }
  
  Matrix(float[] v) {
    h = v.length;
    w = 1;
    pos = new float[h][w];
    for (int i = 0; i < h; i++) {
      pos[i][0] = v[i];
    }
  }
  
  void printAll() {
    for (int i = 0; i < h; i++) {
      println("");
      for (int j = 0; j < w; j++) {
        print(pos[i][j] + " ");
      }
    }
  }
  
  boolean isVector() {
    return w == 1;
  }
  
  void randomize(float from, float to) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        pos[i][j] = random(from, to);
      }
    }
  }
  
  void randomizeRound(float from, float to) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        pos[i][j] = round(random(from, to));
      }
    }
  }
  
  void randomize(float mag) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        pos[i][j] = random(-mag, mag);
      }
    }
  }
  
  Matrix copy() {
    return new Matrix(this);
  }
  
  void set(Matrix x) {
    pos = x.pos;
  }
  
  float[] colAt(int index) {
    float[] x = new float[h];
    for (int i = 0; i < h; i++) {
      x[i] = pos[i][index];
    }
    return x;
  }
  
  Matrix scale(float mult) {
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        pos[i][j] *= mult;
      }
    }
    return this.copy();
  }
  
  Matrix hadamard(Matrix x) {
    if (!(x.w == w && x.h == h)) throw new RuntimeException("The matrices must be of the same dimentions");
    Matrix me = this.copy();
    for (int i = 0; i < h; i++) {
      for (int j = 0; j < w; j++) {
        me.pos[i][j] *= x.pos[i][j];
      }
    }
    return me;
  }
  
  Matrix transpose() {
    Matrix newMatrix = new Matrix(w, h);
    for (int i = 0; i < newMatrix.h; i++) {
      for (int j = 0; j < newMatrix.w; j++) {
        newMatrix.pos[i][j] = pos[j][i];
      }
    }
    return newMatrix;
  }
  
  float dot(Matrix x) {
    if (!(w == 1 && x.w == 1 && h == x.h)) throw new RuntimeException("The matrices must be vectors of the same size");
    float sum = 0;
    for (int i = 0; i < h; i++) {
      sum += pos[i][0] * x.pos[i][0];
    }
    return sum;
  }
  
  Matrix add(Matrix x) {
    if (!(w == x.w && h == x.h)) throw new RuntimeException("matrices must be of same dimentions. w: " + w + " x.w: " + x.w + " h: " + h + " x.h: " + x.h);
    Matrix newMatrix = this.copy();
    for (int i = 0; i < newMatrix.h; i++) {
      for (int j = 0; j < newMatrix.w; j++) {
        newMatrix.pos[i][j] += x.pos[i][j];
      }
    }
    return newMatrix;
  }
  
  Matrix sub(Matrix x) {
    Matrix temp = x.copy();
    return this.add(temp.scale(-1));
  }
  
  Matrix mult(Matrix x) {
    if (w != x.h) throw new RuntimeException("this.w must be equal to x.h");
    Matrix newMatrix = new Matrix(h, x.w);
    for (int i = 0; i < newMatrix.h; i++) {
      for (int j = 0; j < newMatrix.w; j++) {
        Matrix first = new Matrix(pos[i]);
        Matrix seacond = new Matrix(x.colAt(j));
        newMatrix.pos[i][j] = first.dot(seacond);
      }
    }
    return newMatrix;
  }
  
  float sigmoid(float x) {
    return 1 / (1 + pow(e, -x));
  }
  
  float sigmoidPrime(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
  }
  
  Matrix sigmoid() {
    Matrix newMatrix = this.copy();
    for (int i = 0; i < newMatrix.h; i++) {
      for (int j = 0; j < newMatrix.w; j++) {
        newMatrix.pos[i][j] = sigmoid(newMatrix.pos[i][j]);
      }
    }
    return newMatrix;
  }
  
  Matrix sigmoidPrime() {
    Matrix newMatrix = this.copy();
    for (int i = 0; i < newMatrix.h; i++) {
      for (int j = 0; j < newMatrix.w; j++) {
        newMatrix.pos[i][j] = sigmoidPrime(newMatrix.pos[i][j]);
      }
    }
    return newMatrix;
  }
}