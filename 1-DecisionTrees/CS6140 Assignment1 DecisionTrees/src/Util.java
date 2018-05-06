import java.util.List;

/**
 * Utility functions of the project
 * @author Garrett
 *
 */
public class Util {
	/**
	 * Calculate entropy for a ContinuousData list
	 * @param data
	 * @return
	 */
	static double continuousEntropy(List<ContinuousData> data) {
		double entropy = 0;
		int classes = data.get(0).classes;
		int[] counts = new int[classes];
		for (ContinuousData cd : data) {
			counts[cd.y]++;
		}

		for (int i = 0; i < classes; i++) {
			double pqk = (double)counts[i] / (double)data.size();			
			entropy += pqk == 0 ? 0 : -pqk * (Math.log(pqk) / Math.log(2));
		}
		return entropy;

	}

	/**
	 * Calculate entropy for a DiscreteData list
	 * @param data
	 * @return
	 */
	static double discreteEntropy(List<DiscreteData> data) {
		double entropy = 0;

		int classes = DiscreteData.classes();
		int[] counts = new int[classes];
		for (DiscreteData dd : data) {
			counts[dd.y]++;
		}

		for (int i = 0; i < classes; i++) {
			double pqk = (double)counts[i] / (double)data.size();			
			entropy += pqk == 0 ? 0 : -pqk * (Math.log(pqk) / Math.log(2));
		}
		return entropy;

	}

	/**
	 * Calculate the predicted values of a regression data list
	 * @param data
	 * @return
	 */
	static double predictedValue(List<RegressionData> data) {
		double sum = 0;
		for (RegressionData rd : data) {
			sum += rd.y;
		}

		double pv = sum / data.size();

		return pv;
	}

	/**
	 * Calculate the sum of square errors of a regression data list
	 * @param data
	 * @return
	 */
	static double sumOfSquareErrors(List<RegressionData> data) {
		double sse = 0;

		for (RegressionData rd : data) {
			double diff = rd.y - predictedValue(data);
			sse += diff * diff;
		}

		return sse;
	}

	/**
	 * Calculate mean of a double list
	 * @param list
	 * @return
	 */
	static double mean(List<Double> list) {
		double sum = 0;
		for (double d : list) {
			sum += d;
		}

		return sum / list.size();
	}

	/**
	 * Calculate standard deviation of a double list
	 * @param list
	 * @return
	 */
	static double standardDeviation(List<Double> list) {
		double mean = mean(list);

		double sum = 0;
		for (double d : list) {
			sum += (d-mean) * (d-mean);
		}

		return Math.sqrt(sum / list.size());
	}

	/**
	 * Normalize a continuous data list
	 * @param list
	 */
	static void continuousNormalize(List<ContinuousData> list) {
		for (int i = 0; i < list.get(0).x.length; i++) {
			double min = 0;
			double max = 0;

			for (int j = 0; j < list.size(); j++) {
				min = Math.min(min, list.get(j).x[i]);
				max = Math.max(max, list.get(j).x[i]);
			}

			double diff = max - min;
			for (int j = 0; j < list.size(); j++) {
				list.get(j).x[i] = (list.get(j).x[i] - min) / diff;
			}
		}

	}
	
	/**
	 * Normalize a regression data list
	 * @param list
	 */
	static void regressionNormalize(List<RegressionData> list) {
		for (int i = 0; i < list.get(0).x.length; i++) {
			double min = 0;
			double max = 0;

			for (int j = 0; j < list.size(); j++) {
				min = Math.min(min, list.get(j).x[i]);
				max = Math.max(max, list.get(j).x[i]);
			}

			double diff = max - min;
			for (int j = 0; j < list.size(); j++) {
				list.get(j).x[i] = (list.get(j).x[i] - min) / diff;
			}
		}

	}


}

