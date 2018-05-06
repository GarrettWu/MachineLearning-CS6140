
import java.util.Collections;
import java.util.List;

/**
 * Test and print regression data, housing
 * @author Garrett
 *
 */
public class Problem6 {	
	public static void main(String[] args) {
		// read data
		List<RegressionData> housings = RegressionData.read();
		
		Collections.shuffle(housings);
		Util.regressionNormalize(housings);

		//test 
		System.out.println("***Test for regression housing***");
		double[] housingEtas = {0.05, 0.10, 0.15, 0.20};
		for (double eta : housingEtas) {
			RegressionKFoldCrossValidation k = new RegressionKFoldCrossValidation(housings, eta, 10);
			k.validate();
		}
		
		// plot fitness
		double[] etas = new double[30];
		for (int i = 1; i <= etas.length; i++) {
			etas[i-1] = 0.01 * i;
		}
		
		System.out.println("***Test for fitting housing***");
		for (double eta : etas) {
			RegressionKFoldCrossValidation k = new RegressionKFoldCrossValidation(housings, eta, 10);
			k.testFitting();
		}

	}

}