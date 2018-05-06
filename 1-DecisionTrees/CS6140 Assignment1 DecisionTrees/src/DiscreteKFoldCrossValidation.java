import java.util.ArrayList;
import java.util.List;

public class DiscreteKFoldCrossValidation {
	List<DiscreteData> data;
	double eta;
	int k;

	public DiscreteKFoldCrossValidation(List<DiscreteData> data, double eta, int k) {
		this.data = data;
		this.eta = eta;
		this.k = k;
	}

	public void validate() {
		// size of each test
		int size = data.size() / k;

		for (int i = 0; i < k; i++) {
			List<DiscreteData> testing = new ArrayList<DiscreteData>(size);
			List<DiscreteData> training = new ArrayList<DiscreteData>(data.size() - size);

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

			DiscreteDecisionTree tree = new DiscreteDecisionTree(training, eta);
			for (DiscreteData dd : testing) {
				dd.predict = tree.predict(dd);
			}			
		}
		
		printAccuracies();
		printConfusionMatrix();
	}

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

	public void printConfusionMatrix() {
		int classes = DiscreteData.classes();
		
		int[][] matrix = new int[classes][classes];
		
		for (DiscreteData dd : data) {
			matrix[dd.predict][dd.y]++;
		}
		
		System.out.println("Confusion Matrix: Rows: Predict classes, Cols: Actual classes");
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix.length; j++) {
				System.out.print(matrix[i][j] + "  ");
			}
			System.out.println();
		}
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
			List<DiscreteData> testing = new ArrayList<DiscreteData>(size);
			List<DiscreteData> training = new ArrayList<DiscreteData>(data.size() - size);

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

			DiscreteDecisionTree tree = new DiscreteDecisionTree(training, eta);	
			
			int trainingCount = 0, testingCount = 0;
			for (int j = 0; j < training.size(); j++) {
				DiscreteData cd = training.get(j);
				trainingCount += tree.predict(cd) == cd.y ? 1 : 0;
			}
			trainingAccuracies.add((double)trainingCount / (double)training.size());
			
			for (int j = 0; j < testing.size(); j++) {
				DiscreteData cd = testing.get(j);
				testingCount += tree.predict(cd) == cd.y ? 1 : 0;
			}
			
			testingAccuracies.add((double)testingCount / (double)testing.size());
		}
		System.out.format("eta: %.2f    ", eta);
		System.out.format("training accuracy: %.4f    ", Util.mean(trainingAccuracies));
		System.out.format("testing accuracy: %.4f\n", Util.mean(testingAccuracies));
	}
}
