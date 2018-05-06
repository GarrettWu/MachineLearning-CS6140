import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Classification problems with continuous data set, iris and spambase
 * @author Garrett
 *
 */
public class ContinuousData {

	/**
	 * Attributes
	 */
	public double[] x;

	/**
	 * Class
	 */
	public int y;
	
	/**
	 * Record predict value by decision tree
	 */
	public int predict;
	
	/**
	 * Number of possible outputs
	 */
	public int classes;

	/**
	 * Constructor
	 * @param x
	 * @param y
	 * @param classes
	 */
	ContinuousData(double[] x, int y, int classes) {
		this.x = x;
		this.y = y;
		this.classes = classes;
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
	 * Transfer iris classes to int
	 * @param s
	 * @return
	 */
	private static int irisStringToY(String s) {
		return s.equals("Iris-setosa") ? 0 : s.equals("Iris-versicolor") ? 1 : 2;
	}

	/**
	 * Read "iris" or "spam" data set
	 * @param dataType
	 * @return
	 */
	public static List<ContinuousData> read(String dataType) {
		String CSVFILE = null; // file path
		int n = 0; // number of attributes
		
		boolean isIris = false;
		if (dataType.equals("iris")) {
			CSVFILE = "iris.csv";
			n = 4;
			isIris = true;
		} else if (dataType.equals("spam")) {
			CSVFILE = "spambase.csv";
			n = 57;
		}
		
		String line = "";
		String cvsSplitBy = ",";

		List<ContinuousData> list = new ArrayList<ContinuousData>();

		try (BufferedReader br = new BufferedReader(new FileReader(CSVFILE))) {

			while ((line = br.readLine()) != null && !line.equals("")) {
				String[] str = line.split(cvsSplitBy);

				double[] x = new double[n];
				for (int i = 0; i < n; i++ ) {
					x[i] = Double.parseDouble(str[i]);
				}
				
				int y;
				ContinuousData data;
				if (isIris) {
					y = irisStringToY(str[n]);
					data = new ContinuousData(x, y, 3);
				} else {
					y = Integer.parseInt(str[n]);
					data = new ContinuousData(x, y, 2);
				}

				list.add(data);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}

		return list;
	}


}
