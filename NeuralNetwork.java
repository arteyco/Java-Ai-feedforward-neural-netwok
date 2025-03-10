
package neuralnetwork;

import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {    
 
private int inputSize; 
private int hiddenSize;
 private int outputSize; 
private double[][] weightsInputHidden; // Weights from input layer to hidden layer
private double[] biasHidden; // Biases for hidden layer 
private double[][] weightsHiddenOutput; // Weights from hidden layer to output layer 
private double[] biasOutput; // Bias for output layer 
private Random random = new Random(); // For initializing weights and biases 

public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) { 
this.inputSize = inputSize;
this.hiddenSize = hiddenSize; 
this.outputSize = outputSize; // Initialize weights and biases 
weightsInputHidden = new double[inputSize][hiddenSize]; 
biasHidden = new double[hiddenSize]; 
weightsHiddenOutput = new double[hiddenSize][outputSize]; 
biasOutput = new double[outputSize]; 
initializeWeightsAndBiases();
 } 
private void initializeWeightsAndBiases() { // Initialize weights with small random values 
for (int i = 0; i < inputSize; i++) {
 for (int j = 0; j < hiddenSize; j++) {
	weightsInputHidden[i][j] = random.nextGaussian() * 0.1; // Small random values
System.out.print(weightsInputHidden[i][j] + " value of WIH "); 
}
    System.out.println();// New line after each row
 } 
 System.out.println();// New line after each row

for (int i = 0; i < hiddenSize; i++) { 
biasHidden[i] = random.nextGaussian() * 0.1;
    System.out.println("value of bH: " + biasHidden[i]);
}
 System.out.println();// New line after each row
 
 for (int i = 0; i < hiddenSize; i++) { 
for (int j = 0; j < outputSize; j++) { weightsHiddenOutput[i][j] = random.nextGaussian() * 0.1; 
System.out.print(weightsHiddenOutput[i][j] + " value of WHO");
}
   System.out.println(); // New line after each row
 }
 for (int i = 0; i < outputSize; i++) {
 biasOutput[i] = random.nextGaussian() * 0.1;
System.out.print(biasOutput[i] + " value of BO"); 
}
    System.out.println();// New line after each row
   }
 // Sigmoid activation function 
private double sigmoid(double x) {
    double sig = 1 / (1 + Math.exp(-x)); 
    System.out.println("X : " + x);
    System.out.println("Sig: " + sig);
    return 1 / (1 + Math.exp(-x));
 
    
} 


// Forward propagation 
public double[] predict(double[] input) {
 // Hidden layer 
double[] hiddenActivations = new double[hiddenSize]; 
for (int i = 0; i < hiddenSize; i++) {
 double weightedSum = 0;

 for (int j = 0; j < inputSize; j++) { 
weightedSum += input[j] * weightsInputHidden[j][i];
    System.out.println("WS: " + weightedSum);
 }
System.out.println();
 weightedSum += biasHidden[i]; 
hiddenActivations[i] = sigmoid(weightedSum); 
// Apply activation function 
System.out.println("WS2: " + weightedSum);
}
    System.out.println();
 // Output layer 
double[] output = new double[outputSize];
 for (int i = 0; i < outputSize; i++) {
 double weightedSum = 0;
 for (int j = 0; j < hiddenSize; j++) {
 weightedSum += hiddenActivations[j] * weightsHiddenOutput[j][i]; 
}
 weightedSum += biasOutput[i];
 output[i] = sigmoid(weightedSum);
 } 
return output; 
} 

/**
     * @param args the command line arguments
     */
public static void main(String[] args) {
 // Example usage
 NeuralNetwork nn = new NeuralNetwork(2, 3, 1); // 2 inputs, 3 hidden nodes, 1 output 
// Sample input
 double[] input = {0.3, 0.8
};
 // Make a prediction
 double[] prediction = nn.predict(input);
 System.out.println("Prediction: " + Arrays.toString(prediction)); // Output the single prediction value 
} 
}    
    
    

