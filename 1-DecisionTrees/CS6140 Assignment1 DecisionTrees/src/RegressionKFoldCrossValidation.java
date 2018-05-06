import java.util.ArrayList;
import java.util.List;

/**
 * K-fold cross validation for regression data set, housing
 * @author Garrett
 *
 */
public class RegressionKFoldCrossValidation {
	/**
	 * Input data
	 */
	List<RegressionData> data;
	
	/**
	 * Eta to determin when to stop
	 */
	double eta;
	
	/**
	 * K-fold
	 */
	int k;

	/**
	 * Constructor
	 * @param data
	 * @param eta
	 * @param k
	 */
	public RegressionKFoldCrossValidation(List<RegressionData> data, double eta, int k) {
		this.data = data;
		this.eta = eta;
		this.k = k;
	}

	/**
	 * Calculate the predict values of the data
	 */
	public void validate() {
		// size of each test
		int size = data.size() / k;

		for (int i = 0; i < k; i++) {
			List<RegressionData> testing = new ArrayList<RegressionData>(size);
			List<RegressionData> training = new ArrayList<RegressionData>(data.size() - size);

			int start = i * size;
			int end = (i+1) * size;
			
			if (i == k-1) {
				end = data.size();
			}

			for (int j = 0; j < data.size(); j++) {
				if (j >= start && j < end) {
					testing.add(data.get(j));
				} else {
					training.add(data.get(j));
				}
			}

			RegressionDecisionTree tree = new RegressionDecisionTree(training, eta);
			for (RegressionData rd : testing) {
				rd.predict = tree.predict(rd);
			}			
		}
		
		printSSEs();
	}

	/**
	 * Calculate and print sum of square errors for the predict value and actual value
	 */
	public void printSSEs() {
		// size of each test
		int size = data.size() / k;
		List<Double> sses = new ArrayList<Double>(k);

		for (int i = 0; i < k; i++) {
			
			int start = i * size;
			int end = (i+1) * size;
			
			if (i == k-1) {
				end = data.size();
			}
			
			double sse = 0;
			for (RegressionData rd : data.subList(start, end)) {
				double diff = rd.y - rd.predict;
				sse += diff * diff;
			}
			
			sses.add(sse);
		}
		double mean = Util.mean(sses);
		double sd = Util.standardDeviation(sses);
		System.out.format("Eta: %.2f  Mean: %.2f  SD: %.2f\n", eta, mean, sd);
	}
	
	/**
	 * Print test accuracies for both training and testing data
	 */
	public void testFitting() {
		// size of each test
		int size = data.size() / k;
		
		List<Double> trainingSSEs = new ArrayList<Double>(k);
		List<Double> testingSSEs = new ArrayList<Double>(k);
		
		for (int i = 0; i < k; i++) {
			List<RegressionData> testing = new ArrayList<RegressionData>(size);
			List<RegressionData> training = new ArrayList<RegressionData>(data.size() - size);

			int start = i * size;
			int end = (i+1) * size;

			if (i == k-1) {
				end = data.size();
			}

			for (int j = 0; j < data.size(); j++) {
				if (j >= start && j < end) {
					testing.add(data.get(j));
				} else {
					training.add(data.get(j));
				}
			}

			RegressionDecisionTree tree = new RegressionDecisionTree(training, eta);	
			
			double trainingSSE = 0, testingSSE = 0;
			for (int j = 0; j < training.size(); j++) {
				RegressionData cd = training.get(j);
				double diff = tree.predict(cd) - cd.y;
				trainingSSE += diff * diff;
			}
			trainingSSEs.add(trainingSSE);
			
			for (int j = 0; j < testing.size(); j++) {
				RegressionData cd = testing.get(j);
				double diff = tree.predict(cd) - cd.y;
				testingSSE += diff * diff;
			}
			
			testingSSEs.add(testingSSE);
		}
		System.out.format("eta: %.2f    ", eta);
		System.out.format("training average sse: %.4f    ", Util.mean(trainingSSEs) / 9);
		System.out.format("testing average sse: %.4f\n", Util.mean(testingSSEs));
	}
}
