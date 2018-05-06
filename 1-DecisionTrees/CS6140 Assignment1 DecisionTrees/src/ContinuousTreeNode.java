import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.TreeSet;

/**
 * Node of the decision tree for classification continuous data iris and spam
 * @author Garrett
 *
 */
public class ContinuousTreeNode {	
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
	public List<ContinuousData> data;
	
	/**
	 * Split attribute and value of the node
	 */
	public Split split;
	
	/**
	 * Children nodes
	 */
	public ContinuousTreeNode left, right;
	
	/**
	 * Tree instance of the nodes
	 */
	public ContinuousDecisionTree tree;
	
	/**
	 * Vote value
	 */
	int vote = -1;
	
	/**
	 * Entropy
	 */
	double entropy = -1;
	
	/**
	 * Constructor
	 * @param data
	 * @param tree
	 */
	ContinuousTreeNode(List<ContinuousData> data, ContinuousDecisionTree tree) {
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
	 * Size of the data of the node
	 * @return
	 */
	int size() {
		return data.size();
	}
	
	/**
	 * Entropy of the node
	 * @return
	 */
	double entropy() {
		if (entropy == -1) {
			entropy = Util.continuousEntropy(data);
		}
		return entropy;
	}
	
	/**
	 * Information Gain for specific split
	 * @param split
	 * @return
	 */
	double informationGain(Split split) {
		double hq = entropy();
		List<ContinuousData> dataLeft = new ArrayList<ContinuousData>();
		List<ContinuousData> dataRight = new ArrayList<ContinuousData>();
		
		for (ContinuousData i : data) {
			if (i.x[split.feature] <= split.value) {
				dataLeft.add(i);
			} else {
				dataRight.add(i);
			}
		}
		
		double hl = Util.continuousEntropy(dataLeft) * dataLeft.size() / size();
		double hr = Util.continuousEntropy(dataRight) * dataRight.size() / size();
		
		return hq - hl - hr;
	}
	
	/**
	 * Calculate to get the split to maximize information gain
	 * @return
	 */
	public Split getSplit() {
		// if meet stop conditions
		if (entropy() == 0 || data.size() <= tree.minSize) {
			return null;
		}
		
		double maxIG = 0;
		Split maxSplit = null;
		
		// finde the spit of max information gain
		for (int f = 0; f < data.get(0).x.length; f++) {
			// use set to record and sort values
			TreeSet<Double> set = new TreeSet<Double>();
			for (ContinuousData cd : data) {
				set.add(cd.x[f]);
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
				double ig = informationGain(s);
				
				if (ig > maxIG) {
					maxIG = ig;
					maxSplit = s;
				}
			}
		}
		
		// spit the node according to max split		
		if (maxSplit != null) {
			List<ContinuousData> dataLeft = new ArrayList<ContinuousData>();
			List<ContinuousData> dataRight = new ArrayList<ContinuousData>();
			
			for (ContinuousData cd : data) {
				if (cd.x[maxSplit.feature] <= maxSplit.value) {
					dataLeft.add(cd);
				} else {
					dataRight.add(cd);
				}
			}
			
			if (dataLeft.size() != 0 && dataRight.size() != 0) {
				this.left = new ContinuousTreeNode(dataLeft, tree);
				this.right = new ContinuousTreeNode(dataRight, tree);
			}
		}
		
		return maxSplit;
	}
	
	/**
	 * Get the vote value (the most output) of the node
	 * @return
	 */
	public int getVote() {
		if (vote != -1) {
			return vote;
		}
		
		int classes = data.get(0).classes;
		int[] counts = new int[classes];
		for (ContinuousData cd : data) {
			counts[cd.y]++;
		}
		int max = 0, maxI = 0;
		for (int i = 0; i < classes; i++) {
			if (counts[i] > max) {
				maxI = i;
				max = counts[i];
			}
		}
		
		return maxI;
	}
}
