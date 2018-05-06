import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeSet;

/**
 * Tree node of regression tree
 * @author Garrett
 *
 */
public class RegressionTreeNode {
	/**
	 * The splitting feature and threshold of the node
	 * @author Garrett
	 *
	 */
	public static class Split {
		public int feature;
		public double value;

		Split(int feature, double value) {
			this.feature = feature;
			this.value = value;
		}
	}
	/**
	 * List of data
	 */
	public List<RegressionData> data;

	/**
	 * Split attribute and value of the node
	 */
	public Split split;

	/**
	 * Children nodes
	 */
	public RegressionTreeNode left, right;

	/**
	 * Tree of the nodes
	 */
	public RegressionDecisionTree tree;

	/**
	 * Predicted value (mean of the values)
	 */
	double predictedValue = -1;
	
	/**
	 * Sum of square errors
	 */
	double sumOfSquareErrors = -1;

	/**
	 * Constructor
	 * @param data
	 * @param tree
	 */
	RegressionTreeNode(List<RegressionData> data, RegressionDecisionTree tree) {
		this.data = data;	
		this.tree = tree;	
		this.split = getSplit();
	}

	/**
	 * Is leaf node
	 * @return
	 */
	boolean isLeaf() {
		return split == null;
	}

	/**
	 * Size of the data
	 * @return
	 */
	int size() {
		return data.size();
	}

	/**
	 * Get the predicted value
	 * @return
	 */
	double predictedValue() {
		if (predictedValue == -1) {
			predictedValue = Util.predictedValue(data);
		}

		return predictedValue;
	}

	/**
	 * Get sum of square errors
	 * @return
	 */
	double sumOfSquareErrors() {
		if (sumOfSquareErrors == -1) {
			sumOfSquareErrors = Util.sumOfSquareErrors(data);
		}
		return sumOfSquareErrors;
	}

	/**
	 * Calculate drop in error for specific split
	 * @param split
	 * @return
	 */
	double dropInError(Split split) {
		double em = sumOfSquareErrors();

		List<RegressionData> dataLeft = new ArrayList<RegressionData>();
		List<RegressionData> dataRight = new ArrayList<RegressionData>();

		for (RegressionData rd : data) {
			if (rd.x[split.feature] <= split.value) {
				dataLeft.add(rd);
			} else {
				dataRight.add(rd);
			}
		}

		double el = Util.sumOfSquareErrors(dataLeft) * dataLeft.size() / size();
		double er = Util.sumOfSquareErrors(dataRight) * dataRight.size() / size();

		return em - el - er;
	}

	/**
	 * Find the split maximize the drop in error
	 * @return
	 */
	public Split getSplit() {
		// if meet stop conditions
		if (data.size() <= tree.minSize) {
			return null;
		}

		double maxDIE = 0;
		Split maxSplit = null;

		// find the max split
		for (int f = 0; f < data.get(0).x.length; f++) {
			// use set to record and sort values
			TreeSet<Double> set = new TreeSet<Double>();
			for (RegressionData rd : data) {
				set.add(rd.x[f]);
			}
			Iterator<Double> itr = set.iterator();

			List<Double> mids = new LinkedList<Double>();
			double prev = 0;
			while (itr.hasNext()) {
				double cur = itr.next();
				mids.add((cur + prev) / 2);
				prev = cur;
			}

			mids.remove(0);
			for (double mid : mids) {
				Split s = new Split(f, mid);
				double die = dropInError(s);

				if (die > maxDIE) {
					maxDIE = die;
					maxSplit = s;
				}
			}
		}

		// if found, split the node
		if (maxSplit != null) {
			List<RegressionData> dataLeft = new ArrayList<RegressionData>();
			List<RegressionData> dataRight = new ArrayList<RegressionData>();

			for (RegressionData rd : data) {
				if (rd.x[maxSplit.feature] <= maxSplit.value) {
					dataLeft.add(rd);
				} else {
					dataRight.add(rd);
				}
			}

			if (dataLeft.size() != 0 && dataRight.size() != 0) {
				this.left = new RegressionTreeNode(dataLeft, tree);
				this.right = new RegressionTreeNode(dataRight, tree);
			}
		}

		return maxSplit;
	}
}
