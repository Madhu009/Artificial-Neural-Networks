package Binary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import Binary.HiddenLayer;
import Jama.Matrix;
public class Network {

	//list of hidden layers 
	List<HiddenLayer> Hidden=new ArrayList<HiddenLayer>();
	Neuron outputNeuron=new Neuron(0,false);
	List<Neuron> inputLayer=new ArrayList<Neuron>();
	
	private static String fileName="C:/Users/Madhu/Desktop/nn.txt";
	
	int inputLayerSize;int hiddenLayerSize;int noOfHiddenlayers;
	Matrix input;
	Matrix output;
	double learningRate=0.9f;
	private double mommentum=0.1f;
	public Network(int inputLayerSize,int hiddenLayerSize,int noOfHiddenlayers)
	{
		this.inputLayerSize=inputLayerSize;
		this.hiddenLayerSize=hiddenLayerSize;
		this.noOfHiddenlayers=noOfHiddenlayers;
		Matrix data=readMatrix(fileName);
		input=getInput(data);
		output=getOutput(data);
		createNetwork();
		
	}
	
	private void createNetwork()
	{
		//for hiddenlayers 
		//an object holds list of neurons (hiddenlayer)
		//and a list holds list of those objects(list of hidden layers)
		for(int n=0;n<noOfHiddenlayers;n++)
		{
			//check if its the last hidden layer
			//True-->set weightLength =1
			if(n==noOfHiddenlayers-1)
			{
				//obj holds list of neurons
				HiddenLayer hiddenLayer=new HiddenLayer();
				
				Neuron biasNeuron=new Neuron(1,true);
				hiddenLayer.list.add(biasNeuron);
				for(int i=0;i<hiddenLayerSize;i++)
				{
					Neuron hiddenNeuron=new Neuron(1,false);
					hiddenLayer.list.add(hiddenNeuron);
				}
				Hidden.add(hiddenLayer);
			}
			
			//False-->set weightLength=hiddenLayenSize
			else
			{
				//obj holds list of neurons
				HiddenLayer hiddenLayer=new HiddenLayer();
				
				Neuron biasNeuron=new Neuron(hiddenLayerSize,true);
				hiddenLayer.list.add(biasNeuron);
				for(int i=0;i<hiddenLayerSize;i++)
				{
					Neuron hiddenNeuron=new Neuron(hiddenLayerSize,false);
					hiddenLayer.list.add(hiddenNeuron);
				}
				Hidden.add(hiddenLayer);
			}
			
		}
		
		//for inputlayer
		Neuron inputBiasNeuron=new Neuron(hiddenLayerSize,true);
		inputLayer.add(inputBiasNeuron);
		for(int i=0;i<inputLayerSize;i++)
		{
			Neuron inputNeuron=new Neuron(hiddenLayerSize,false);
			inputLayer.add(inputNeuron);
		}
		
	}


	//gives input data matrix
	private Matrix getInput(Matrix data_set) {
		return data_set.getMatrix(0, data_set.getRowDimension() - 1, 0, data_set.getColumnDimension() - 2);
		
		}
		
	//gives output data matrix
	private Matrix getOutput(Matrix data_set) {
		    return data_set.getMatrix(0, data_set.getRowDimension() - 1, data_set.getColumnDimension() - 1, data_set.getColumnDimension() - 1);
		}

	//gives the data from file
	private Matrix readMatrix(String fileName) {
			try {
				BufferedReader reader = new BufferedReader(new FileReader(fileName));
				List<double[]> data_array = new ArrayList<double[]>();

				String line;
				while ((line = reader.readLine()) != null) {
					String fields[] = line.split(",");
					double data[] = new double[fields.length];
					for (int i = 0; i < fields.length; ++i) {
						data[i] = Double.parseDouble(fields[i]);
					}
					data_array.add(data);
				}

				if (data_array.size() > 0) {
					int cols = data_array.get(0).length;
					int rows = data_array.size();
					Matrix matrix = new Matrix(rows, cols);
					for (int r = 0; r < rows; ++r) {
						for (int c = 0; c < cols; ++c) {
							matrix.set(r, c, data_array.get(r)[c]);
						}
					}
					return matrix;
				}
			} catch (Exception e) {
				e.printStackTrace();
				System.exit(1);
			}

		    return new Matrix(0, 0);
		}
	
	
	public static void main(String args[])
	{
		Network net=new Network(2,3,2);
		for(int i=0;i<net.input.getRowDimension();i++)
		{
			double[] input=new double[net.input.getColumnDimension()];
			for(int j=0;j<input.length;j++)
			{
				input[j]=net.input.get(i, j);
			}
			net.forwardPropagation(input);
		}
		
		
	}
	
	//pass the training example to the network
	public void forwardPropagation(double[] input)
	{
		//setting up the input to the input layer
		for(int i=0;i<input.length;i++)
		{
			//exculding bias so start from i+1
			inputLayer.get(i+1).value=input[i];
		}
		//double hiddenToOutputResul0;
		//calculate hiddenlayer from input layer
		for(int j=0;j<hiddenLayerSize;j++)
		{
			double weightsResult=0;
			
			//multiply weghts and inputs
			for(int i=0;i<inputLayerSize;i++)
			{
				weightsResult+=input[i]*inputLayer.get(i+1).weights[j];
			}
			//add bias weight cause value=1
			weightsResult+=inputLayer.get(0).weights[j];
			
			//store the value in hidden neuron j in the list 
			Hidden.get(0).list.get(j+1).value=weightsResult;
			Hidden.get(0).list.get(j+1).output=getSigmoid(weightsResult);
		}
		
		//calculate hidden layer from hidden layer
		if(noOfHiddenlayers>1)
		{
			//repeat for everyhidden layer
			for(int layerlevel=0;layerlevel<noOfHiddenlayers-1;layerlevel++)
			{				
				//repeat for every neuron on that level
				for(int i=0;i<hiddenLayerSize;i++)
				{
					double outputHiddenResult=0;
					//repeat for every neuron
					for(int j=0;j<hiddenLayerSize;j++)
					{
						//hidden layer weight * value
						//j+1 because 1st neuron is bias
						outputHiddenResult+=Hidden.get(layerlevel).list.get(j+1).value*Hidden.get(layerlevel).list.get(j+1).weights[i];
					}
					//add bias of hidden layer i
					outputHiddenResult+=Hidden.get(layerlevel).list.get(0).value*Hidden.get(layerlevel).list.get(0).weights[i];
				
				//store the result in next layer(i+1) neuron(i) 
				Hidden.get(layerlevel+1).list.get(i+1).value=outputHiddenResult;
				Hidden.get(layerlevel+1).list.get(i+1).output=getSigmoid(outputHiddenResult);
				}
			}
		}
		
		//calculate last hidden layer to output values
		double outputValue=0;
		
		for(int i=0;i<hiddenLayerSize;i++)
		{
			//calculate outputWeight(length is 1) * value
			outputValue+=Hidden.get(noOfHiddenlayers-1).list.get(i+1).value*Hidden.get(noOfHiddenlayers-1).list.get(i+1).OutputWeight;
		}
		//add bias
		outputValue+=Hidden.get(noOfHiddenlayers-1).list.get(0).value*Hidden.get(noOfHiddenlayers-1).list.get(0).OutputWeight;
		
		//store the value in output neuron
		outputNeuron.value=outputValue;
		outputNeuron.output=getSigmoid(outputValue);
		
	}
	
	
	public void backPropagation(double expected)
	{
		double error=0;
		error=expected-outputNeuron.output-expected;
		
		//calculate output neuron delta output
		//so first sigmoid inverse of output layer
		double sigmoidInverse=(1-getSigmoid(outputNeuron.value))*getSigmoid(outputNeuron.output);
		double deltaOutputValue=sigmoidInverse*error;
		//calculate hiddenSum for every neuron
		double[] hiddenValue=new double[hiddenLayerSize+1];
		//calculate weights for last hidden layer
		for(int i=0;i<=hiddenLayerSize;i++)
		{
			double deltaHiddenWeight=0;
			
			//deltaWeight= learningrate*deltaOutputValue*HiddenOutput
			deltaHiddenWeight=learningRate*deltaOutputValue*Hidden.get(noOfHiddenlayers-1)
					.list.get(i).output;
			
			//oldweight+newweight=hiddenweight
			double newWeight=Hidden.get(noOfHiddenlayers-1)
					.list.get(i).OutputWeight+deltaHiddenWeight;
			
			//set new weight with mommentum
			Hidden.get(noOfHiddenlayers-1)
			.list.get(i).OutputWeight=newWeight+(deltaHiddenWeight*mommentum);
			
			//hiddenweight* outputLayeroutput*sigmoidInverseHiddenValue=
			//hiddenValue
			
			double sigInverseHiddenValue=getSigmoid(Hidden.get(noOfHiddenlayers-1)
					.list.get(i).value) * (1-getSigmoid(Hidden.get(noOfHiddenlayers-1)
					.list.get(i).value));
			
			hiddenValue[i]=Hidden.get(noOfHiddenlayers-1)
					.list.get(i).OutputWeight*deltaOutputValue*sigInverseHiddenValue;
		}
		
		
	}
	
	private double getSigmoid(double hyp)
	{
		 return 1.0/(1+Math.exp(-hyp));
	}
	
	
	
}
