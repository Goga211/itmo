package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Ln;
import ru.traphouse.functions.log.Log5;

import java.util.HashMap;
import java.util.Map;

public class Log5Stub extends Log5 {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(1.0,    0.0);
        TABLE.put(5.0,    1.0);
        TABLE.put(25.0,   2.0);
        TABLE.put(0.2,   -1.0);
        TABLE.put(2.0,    0.43067655807339306);
        TABLE.put(10.0,   1.4306765580733931);
    }

    public Log5Stub(Ln ln) {
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
