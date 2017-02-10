package Binary;

import java.util.Random;

public class Neuron {

	// Activated fuction output(sigmoid)
	double output;
	
	//product of theta and inputs
	double value;
	
	//count of nodes in the next layer connected to it
	int weightLength;
	
	//theta values
	double[] weights;
	//1 weight for the output
	double OutputWeight=0;
	
	//change value(2nd time onwards)
	double prevDeltaValue=0;
	
	Random r=new Random();
	
	
	//constuctor with weightlength and bias status
	public Neuron(int weightLength,boolean bias)
	{
		this.weightLength=weightLength;
		
		//check if it is not connected to output neuron
		//true-->random weights
		if(weightLength>1)
		{
			weights=new double[weightLength];
			for(int i=0;i<weightLength;i++)
			{
				weights[i]=0.0+(1.0-0.0)*r.nextDouble();
			}
		}
		//false-->Random 1 weight
		else if(weightLength==1)
		{
			OutputWeight=0.0+(1.0-0.0)*r.nextDouble();
		}
		
		//check if the node is bias 
		//true-->
		if(bias==true)
		{
			this.value=1;
			this.output=1;
		}
		
	}
	
	
	
	
	
}
