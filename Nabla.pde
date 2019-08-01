class Nabla {
  Matrix[] w;
  Matrix[] b;
  
  Nabla(Matrix[] w, Matrix[] b) {
    if (w.length != b.length) throw new RuntimeException("weights and biases should be the same size");
    this.w = new Matrix[w.length];
    this.b = new Matrix[b.length];
    for (int i = 0; i < w.length; i++) {
      this.w[i] = w[i] == null ? null : w[i].copy();
      this.b[i] = b[i] == null ? null : b[i].copy();
    }
  }
  
  Nabla copy() {
    return new Nabla(w, b);
  }
  
  void add(Nabla x) {
    if (w.length != x.w.length) throw new RuntimeException("should add with a nabla of same size");
    for (int i = 0; i < w.length; i++) {
      if (w[i] != null) if (w[i].w != x.w[i].w || w[i].h != x.w[i].h) throw new RuntimeException("should add with a nabla of same shapes");
      if (w[i] != null) {
        w[i] = w[i].add(x.w[i]);
      }
      if (b[i] != null) {
        b[i] = b[i].add(x.b[i]);
      }
    }
  }
  
  void printData() {
    for (int i = 1; i < w.length; i++) {
      println("\n\nWeights at layer " + i);
      w[i].printAll();
      println("\n\nBiases at layer " + i);
      b[i].printAll();
    }
  }
  
  Nabla scale(float mult) {
    Nabla newNabla = this.copy();
    for (int i = 0; i < w.length; i++) {
      newNabla.w[i].scale(mult);
      newNabla.b[i].scale(mult);
    }
    return newNabla;
  }
}

class Batch {
  Matrix[] x;
  Matrix[] y;
  int size;
  
  Batch(Matrix[] x, Matrix[] y) {
    if (x.length != y.length) throw new RuntimeException("input data and output data should be of the same size");
    if (!x[0].isVector())throw new RuntimeException("input data must be vectors");
    if (!y[0].isVector())throw new RuntimeException("output data must be vectors");
    this.x = x;
    this.y = y;
    size = x.length;
  }
}