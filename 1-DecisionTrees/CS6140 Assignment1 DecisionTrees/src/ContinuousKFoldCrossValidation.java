import java.util.ArrayList;
import java.util.List;

/**
 * Implement k-fold cross validation for classification continuous data
 * @author Garrett
 *
 */
public class ContinuousKFoldCrossValidation {
	/**
	 * The data set
	 */
	List<ContinuousData> data;

	/**
	 * Eta value to stop split
	 */
	double eta;

	/**
	 * k-fold
	 */
	int k;

	/**
	 * Constructor
	 * @param data
	 * @param eta
	 * @param k
	 */
	public ContinuousKFoldCrossValidation(List<ContinuousData> data, double eta, int k) {
		this.data = data;
		this.eta = eta;
		this.k = k;
	}

	/**
	 * Calculate predict values for the data set using continuous decision tree,
	 * then print accuracies.
	 */
	public void validate() {
		// size of each test
		int size = data.size() / k;

		for (int i = 0; i < k; i++) {
			List<ContinuousData> testing = new ArrayList<ContinuousData>(size);
			List<ContinuousData> training = new ArrayList<ContinuousData>(data.size() - size);

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

			ContinuousDecisionTree tree = new ContinuousDecisionTree(training, eta);
			for (ContinuousData cd : testing) {
				cd.predict = tree.predict(cd);
			}			
		}
		printAccuracies();
		printConfusionMatrix();
	}

	/**
	 * Print the mean and standard deviation of the accuracies
	 */
	public void printAccuracies() {
		// size of each test
		int size = data.size() / k;
		List<Double> accuracies = new ArrayList<Double>(k);

		for (int i = 0; i < k; i++) {
			int count = 0;
			int start = i * size;
			int end = (i+1) * size;
			if (i == k-1) {
				end = data.size();
			}

			for (int j = start; j < end; j++) {
				count += data.get(j).predict == data.get(j).y ? 1 : 0; 
			}

			accuracies.add((double)count / (double)size);
		}

		double mean = Util.mean(accuracies);
		double sd = Util.standardDeviation(accuracies);
		System.out.format("Eta: %.2f  Mean: %.4f  SD: %.4f\n", eta, mean, sd);
	}

	/**
	 * Print confusion matrix of the predicts
	 */
	public void printConfusionMatrix() {
		int classes = data.get(0).classes;

		int[][] matrix = new int[classes][classes];

		for (ContinuousData cd : data) {
			matrix[cd.predict][cd.y]++;
		}

		System.out.println("Confusion Matrix: Rows: Predict classes, Cols: Actual classes");
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				System.out.print(matrix[i][j] + "  ");
			}
			System.out.println();
		}
		System.out.println();
	}

	/**
	 * Print test accuracies for both training and testing data
	 */
	public void testFitting() {
		// size of each test
		int size = data.size() / k;
		
		List<Double> trainingAccuracies = new ArrayList<Double>(k);
		List<Double> testingAccuracies = new ArrayList<Double>(k);
		
		for (int i = 0; i < k; i++) {
			List<ContinuousData> testing = new ArrayList<ContinuousData>(size);
			List<ContinuousData> training = new ArrayList<ContinuousData>(data.size() - size);

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

			ContinuousDecisionTree tree = new ContinuousDecisionTree(training, eta);	
			
			int trainingCount = 0, testingCount = 0;
			for (int j = 0; j < training.size(); j++) {
				ContinuousData cd = training.get(j);
				trainingCount += tree.predict(cd) == cd.y ? 1 : 0;
			}
			trainingAccuracies.add((double)trainingCount / (double)training.size());
			
			for (int j = 0; j < testing.size(); j++) {
				ContinuousData cd = testing.get(j);
				testingCount += tree.predict(cd) == cd.y ? 1 : 0;
			}
			
			testingAccuracies.add((double)testingCount / (double)testing.size());
		}
		System.out.format("eta: %.2f    ", eta);
		System.out.format("training accuracy: %.4f    ", Util.mean(trainingAccuracies));
		System.out.format("testing accuracy: %.4f\n", Util.mean(testingAccuracies));
	}
}
