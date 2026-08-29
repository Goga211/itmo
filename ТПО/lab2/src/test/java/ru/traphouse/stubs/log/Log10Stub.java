package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Ln;
import ru.traphouse.functions.log.Log10;

import java.util.HashMap;
import java.util.Map;

public class Log10Stub extends Log10 {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(1.0,    0.0);
        TABLE.put(10.0,   1.0);
        TABLE.put(100.0,  2.0);
        TABLE.put(0.1,   -1.0);
        TABLE.put(2.0,    0.3010299957316877);
        TABLE.put(5.0,    0.6989700042683123);
    }

    public Log10Stub(Ln ln) {
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
