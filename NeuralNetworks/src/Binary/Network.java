package Binary;

import java.util.ArrayList;
import java.util.List;

import Binary.HiddenLayer;
public class Network {

	//list of hidden layers 
	List<HiddenLayer> Hidden=new ArrayList<HiddenLayer>();
	Neuron outputNeuron=new Neuron(0,false);
	List<Neuron> inputLayer=new ArrayList<Neuron>();
	
	public void createNetwork(int inputLayerSize,int hiddenLayerSize,int noOfHiddenlayers)
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
	
}
