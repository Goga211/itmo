package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Ln;
import ru.traphouse.functions.log.Log2;

import java.util.HashMap;
import java.util.Map;

public class Log2Stub extends Log2 {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(1.0,   0.0);
        TABLE.put(2.0,   1.0);
        TABLE.put(4.0,   2.0);
        TABLE.put(8.0,   3.0);
        TABLE.put(0.5,  -1.0);
        TABLE.put(3.0,   1.5849625007211563);
        TABLE.put(5.0,   2.321928094887362);
    }

    public Log2Stub(Ln ln) {
        super(ln);
    }

    @Override
    public double calculate(double x) {
        for (Map.Entry<Double, Double> entry : TABLE.entrySet()) {
            if (Math.abs(entry.getKey() - x) < 1e-9) {
                return entry.getValue();
            }
        }
        return super.calculate(x);
    }
}
