import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Discrete data type for mushroom
 * @author Garrett
 *
 */
public class DiscreteData {

	/**
	 * Attributes
	 */
	public int[] x;

	/**
	 * Map characters to int indices
	 */
	public static Map<Character, Integer>[] maps;
	
	/**
	 * Class
	 */
	public int y;
	
	/**
	 * Record predict value
	 */
	public int predict;
	
	/**
	 * If the data has been transfered to binary form
	 */
	public boolean isBinary = false;
	
	/**
	 * Constructor
	 * @param x
	 * @param y
	 */
	DiscreteData(int[] x, int y) {
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
	 * The number of possible outputs
	 * @return
	 */
	public static int classes() {
		return 2;
	}
	
	/**
	 * Get number of possible values for a specific feature
	 * @param feature
	 * @return
	 */
	public int getFeatureSize(int feature) {
		if (isBinary) {
			return 2;
		} else {
			return maps[feature].size();
		}
	}
	
	/**
	 * Read the data file into a list
	 * @return
	 */
	@SuppressWarnings("unchecked")
	public static List<DiscreteData> read() {
		String CSVFILE = "mushroom.csv"; // file path
		int n = 21; // number of attributes
		maps = new HashMap[n];
		for (int i = 0; i < n; i++) {
			Map<Character, Integer> map = new HashMap<Character, Integer>();
			maps[i] = map;
		}
		
		String line = "";
		String cvsSplitBy = ",";

		List<DiscreteData> list = new ArrayList<DiscreteData>();

		try (BufferedReader br = new BufferedReader(new FileReader(CSVFILE))) {

			while ((line = br.readLine()) != null && !line.equals("")) {
				String[] str = line.split(cvsSplitBy);

				int[] x = new int[n];
				for (int i = 0; i < n; i++ ) {
					char c = str[i].charAt(0);
					
					if (maps[i].containsKey(c)) {
						x[i] = maps[i].get(c);
					} else {
						x[i] = maps[i].size();
						maps[i].put(c, maps[i].size());
					}
				}
				
				
				char c = str[n].charAt(0);
				int y = c == 'e' ? 0 : 1;
				
				DiscreteData data = new DiscreteData(x, y);

				list.add(data);
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return list;
	}
	
	/**
	 * Transfer a multiway list to binary form
	 * @param list
	 * @return
	 */
	public static List<DiscreteData> transferToBinary(List<DiscreteData> list) {
		List<DiscreteData> bList = new ArrayList<DiscreteData>(list.size());
		
		int xSize = 0;
		for (Map<Character, Integer> map : maps) {
			xSize += map.size();
		}
		for (DiscreteData mData : list) {
			int[] x  = new int[xSize];
			int count = 0;
			for (int i = 0; i < maps.length; i++) {
				for (int j = 0; j < maps[i].size(); j++) {
					x[count] = mData.x[i] == j ? 1 : 0;
					count++;
				}
			}
			
			DiscreteData bData = new DiscreteData(x, mData.y);
			bData.isBinary = true;
			bList.add(bData);
		}
		
		
		return bList;
	}


}
