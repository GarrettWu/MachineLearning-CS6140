
import java.util.Collections;
import java.util.List;

/**
 * Test and print problem1, classification continuous data, iris and spam
 * @author Garrett
 *
 */
public class Problem1 {	
	public static void main(String[] args) {
		//read data
		List<ContinuousData> irises = ContinuousData.read("iris");
		Collections.shuffle(irises);
		Util.continuousNormalize(irises);

		List<ContinuousData> spams = ContinuousData.read("spam");
		Collections.shuffle(spams);
		Util.continuousNormalize(spams);

		//test
		System.out.println("***Test for data set iris***");
		double[] irisEtas = {0.05, 0.10, 0.15, 0.20};
		for (double eta : irisEtas) {
			ContinuousKFoldCrossValidation k = new ContinuousKFoldCrossValidation(irises, eta, 10);
			k.validate();
		}
//		// spambase runs slow, comment them off for test
//		System.out.println("***Test for data set spam***");
//		double[] spamEtas = {0.05, 0.10, 0.15, 0.20, 0.25};
//		for (double eta : spamEtas) {
//			ContinuousKFoldCrossValidation k = new ContinuousKFoldCrossValidation(spams, eta, 10);
//			k.validate();
//		}
//		
	
		// fitness
		double[] etas = new double[30];
		for (int i = 1; i <= etas.length; i++) {
			etas[i-1] = 0.01 * i;
		}
		
		System.out.println("***Test for fitting iris***");
		for (double eta : etas) {
			ContinuousKFoldCrossValidation k = new ContinuousKFoldCrossValidation(irises, eta, 10);
			k.testFitting();
		}
		
//		System.out.println("***Test for fitting spam***");
//		for (double eta : etas) {
//			ContinuousKFoldCrossValidation k = new ContinuousKFoldCrossValidation(spams, eta, 10);
//			k.testFitting();
//		}
		
	}

}
