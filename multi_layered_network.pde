float inputRange = 1;
float learningRate = 3;
int learningAmount = 2;
int learningCount = 0;

Network net;
float[][] ins;
int[] outs;
int totalData = 0;

void setup() {
  size(1200, 800);
  background(0);
  
  int[] inputSize = {2, 12, 12, 1};
  net = new Network(inputSize);
  ins = new float[0][];
  outs = new int[0];
  showCurve();
}

void showCurve() {
  float size = 10;
  for (int i = 0; i < width / size; i++) {
    for (int j = 0; j < height / size; j++) {
      float x = map(i, 0, width / size, -inputRange, inputRange);
      float y = map(j, 0, height / size, -inputRange, inputRange);
      float[] input = {x, y};
      float a = net.feedForward(new Matrix(input)).pos[0][0];
      color c = color(map(a, 0, 1, 255, 0), map(a, 0, 1, 0, 255), 0);
      fill(c);
      noStroke();
      rect(i * size, j * size, size, size);
    }
  }
  showData();
}

void showData() {
  for (int i = 0; i < totalData; i++) {
    float x = map(ins[i][0], -inputRange, inputRange, 0, width);
    float y = map(ins[i][1], -inputRange, inputRange, 0, height);
    color c = outs[i] == 0 ? color(255, 0, 0) : color(0, 255, 0);
    fill(c);
    stroke(0);
    strokeWeight(2);
    ellipseMode(CENTER);
    ellipse(x, y, 10, 10);
  }
}

void mousePressed() {
  float x = map(mouseX, 0, width, -inputRange, inputRange);
  float y = map(mouseY, 0, height, -inputRange, inputRange);
  float[] newIn = {x, y};
  int newOut = mouseButton == RIGHT ? 0 : 1;
  ins = (float[][]) append(ins, newIn);
  outs = (int[]) append(outs, newOut);
  totalData ++;
  showData();
}

int imgNum = 0;
void saveImg() {
  save("output/img_" + imgNum + ".png");
  imgNum ++;
}

void draw() {
  if (!keyPressed) return;
  if (totalData <= 0) return;
  background(0);
  
  Matrix[] x = new Matrix[totalData];
  Matrix[] y = new Matrix[totalData];
  for (int i = 0; i < totalData; i++) {
    x[i] = new Matrix(ins[i]);
    float[] outsArr = {outs[i]};
    y[i] = new Matrix(outsArr);
  }
  Batch batch = new Batch(x, y);
  
  for (int i = 0; i < learningAmount; i++) net.update_batch(batch, learningRate);
  learningCount += learningAmount;
  
  showCurve();
  saveImg();
  println("learnt: " + learningCount);}
