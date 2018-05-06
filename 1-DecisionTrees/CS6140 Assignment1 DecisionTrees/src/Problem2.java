import java.util.Collections;
import java.util.List;

/**
 * Test and print classification discrete data, mushroom
 * @author Garrett
 *
 */
public class Problem2 {	
	public static void main(String[] args) {
		// read
		List<DiscreteData> mushrooms = DiscreteData.read();
		
		Collections.shuffle(mushrooms);

		// test
		System.out.println("***Test for multiway attributes***");
		double[] mushroomEtas = {0.05, 0.10, 0.15};
		for (double eta : mushroomEtas) {
			DiscreteKFoldCrossValidation k = new DiscreteKFoldCrossValidation(mushrooms, eta, 10);
			k.validate();
		}
		
		System.out.println();
		System.out.println("***Test for binary attributes***");
		List<DiscreteData> mushroomsBinary = DiscreteData.transferToBinary(mushrooms);
		for (double eta : mushroomEtas) {
			DiscreteKFoldCrossValidation k = new DiscreteKFoldCrossValidation(mushroomsBinary, eta, 10);
			k.validate();
		}
		
		// fitness
		double[] etas = new double[30];
		for (int i = 1; i <= etas.length; i++) {
			etas[i-1] = 0.01 * i;
		}
		
		System.out.println("***Test for fitting mushrooms multiway***");
		for (double eta : etas) {
			DiscreteKFoldCrossValidation k = new DiscreteKFoldCrossValidation(mushrooms, eta, 10);
			k.testFitting();
		}
		
		System.out.println("***Test for fitting mushrooms binary***");
		for (double eta : etas) {
			DiscreteKFoldCrossValidation k = new DiscreteKFoldCrossValidation(mushroomsBinary, eta, 10);
			k.testFitting();
		}

	}

}