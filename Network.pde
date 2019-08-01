class Network {
  int numLayers;
  int lastLayer;
  int[] sizes;
  Matrix[] biases;
  Matrix[] weights;
  
  Network(int[] sizes) {
    this.sizes = sizes;
    numLayers = sizes.length;
    lastLayer = numLayers - 1;
    
    biases = new Matrix[numLayers];
    weights = new Matrix[numLayers];
    
    for (int i = 1; i < numLayers; i++) {
      biases[i] = new Matrix(sizes[i]);
      weights[i] = new Matrix(sizes[i], sizes[i-1]);
      
      biases[i].randomize(1);
      weights[i].randomize(1);
    }
  }
  
  void printData() {
    for (int i = 1; i < numLayers; i++) {
      println("\n\nWeights at layer " + i);
      weights[i].printAll();
      println("\n\nBiases at layer " + i);
      biases[i].printAll();
    }
  }
  
  Matrix feedForward(Matrix input) {
    if (!input.isVector()) throw new RuntimeException("input must be a vector");
    if (input.h != sizes[0]) throw new RuntimeException("input must be same size as first neurons");
    Matrix a = input.copy();
    for (int i = 1; i < numLayers; i++) {
      Matrix w = weights[i];
      Matrix b = biases[i];
      a = w.mult(a).add(b).sigmoid();
    }
    return a;
  }
  
  void update_batch(Batch batch, float eta) {
    Matrix[] nablaW = new Matrix[numLayers];
    Matrix[] nablaB = new Matrix[numLayers];
    for (int i = 1; i < numLayers; i++) {
      nablaB[i] = new Matrix(sizes[i]);
      nablaW[i] = new Matrix(sizes[i], sizes[i-1]);
      nablaB[i].scale(0);
      nablaW[i].scale(0);
    }
    Nabla nabla = new Nabla(nablaW, nablaB);
    for (int i = 0; i < batch.size; i++) {
      backprop(batch.x[i], batch.y[i]);
      nabla.add(backprop(batch.x[i], batch.y[i]));
    }
    for (int i = 1; i < numLayers; i++) {
      weights[i] = weights[i].sub(nabla.w[i].scale(eta / batch.size));
      biases[i] = biases[i].sub(nabla.b[i].scale(eta / batch.size));
    }
  }
  
  Nabla backprop(Matrix x, Matrix y) {
    if (!x.isVector()) throw new RuntimeException("input must be a vector");
    if (x.h != sizes[0]) throw new RuntimeException("input must be same size as first neurons");
    if (!y.isVector()) throw new RuntimeException("output must be a vector");
    if (y.h != sizes[lastLayer]) throw new RuntimeException("output must be same size as last neurons, last neurons num: " + sizes[lastLayer] + " output: " + y.h);
    
    Matrix[] nablaW = new Matrix[numLayers];
    Matrix[] nablaB = new Matrix[numLayers];
    
    Matrix activation = x.copy();
    Matrix[] activations = {activation};
    Matrix[] zs = {new Matrix(0)};
    
    for (int i = 1; i < numLayers; i++) {
      Matrix w = weights[i];
      Matrix b = biases[i];
      
      Matrix z = w.mult(activation).add(b);
      activation = z.sigmoid();
      
      zs = (Matrix[]) append(zs, z);
      activations = (Matrix[]) append(activations, activation);
    }
    
    Matrix delta = costDerivadive(activations[lastLayer], y).hadamard(zs[lastLayer].sigmoidPrime());
    nablaB[lastLayer] = delta.copy();
    nablaW[lastLayer] = delta.mult(activations[lastLayer-1].transpose());
    
    for (int l = lastLayer - 1; l >= 1; l--) {
      Matrix z = zs[l];
      Matrix sp = z.sigmoidPrime();
      delta = weights[l+1].transpose().mult(delta).hadamard(sp);
      nablaB[l] = delta.copy();
      nablaW[l] = delta.mult(activations[l-1].transpose());
    }
    
    return new Nabla(nablaW, nablaB);
  }
  
  Matrix costDerivadive(Matrix output, Matrix y) {
    return output.sub(y);
  }
}