import java.util.ArrayList;
import java.util.List;

/**
 * The "golden"/fixed counterpart of SumAdapterBuggy -- correctly sums all
 * values, including negatives. Deliberately self-contained (no reference to
 * SumAdapterBuggy) since a real Defects4J fixed checkout is compiled
 * entirely separately from the buggy one.
 */
public class SumAdapterFixed implements BugAdapter {
    @Override
    public String invoke(String csvInput) throws Exception {
        List<Integer> values = parse(csvInput);
        int sum = 0;
        for (int v : values) {
            sum += v;
        }
        return String.valueOf(sum);
    }

    static List<Integer> parse(String csvInput) {
        List<Integer> values = new ArrayList<>();
        if (csvInput == null || csvInput.isEmpty()) {
            return values;
        }
        for (String part : csvInput.split(",")) {
            values.add(Integer.parseInt(part.trim()));
        }
        return values;
    }
}