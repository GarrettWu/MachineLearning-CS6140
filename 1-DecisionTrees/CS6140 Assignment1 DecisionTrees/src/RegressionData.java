import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Data type for regression tree, housing
 * @author Garrett
 *
 */
public class RegressionData {

	/**
	 * Attributes
	 */
	public double[] x;

	/**
	 * Class
	 */
	public double y;

	/**
	 * Record predict value
	 */
	public double predict;

	/**
	 * Contructor
	 * @param x
	 * @param y
	 */
	RegressionData(double[] x, double y) {
		this.x = x;
		this.y = y;
	}

	/**
	 * To string for test
	 */
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < x.length; i++) {
			sb.append(x[i]);
			sb.append(',');
		}

		sb.append(y);

		return sb.toString();
	}

	/**
	 * Read the data file to list
	 * @return
	 */
	public static List<RegressionData> read() {
		String CSVFILE = "housing.csv"; // file path
		int n = 13; // number of attributes

		String line = "";
		String cvsSplitBy = ",";

		List<RegressionData> list = new ArrayList<RegressionData>();

		try (BufferedReader br = new BufferedReader(new FileReader(CSVFILE))) {

			while ((line = br.readLine()) != null && !line.equals("")) {
				String[] str = line.split(cvsSplitBy);

				double[] x = new double[n];
				for (int i = 0; i < n; i++ ) {
					x[i] = Double.parseDouble(str[i]);
				}

				double y = Double.parseDouble(str[n]);
				RegressionData data = new RegressionData(x, y);

				list.add(data);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

		return list;
	}


}
