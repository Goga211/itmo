package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Ln;
import ru.traphouse.functions.log.Log3;

import java.util.HashMap;
import java.util.Map;

public class Log3Stub extends Log3 {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(1.0,   0.0);
        TABLE.put(3.0,   1.0);
        TABLE.put(9.0,   2.0);
        TABLE.put(27.0,  3.0);
        TABLE.put(2.0,   0.6309297535714573);
        TABLE.put(5.0,   1.4649735207179273);
    }

    public Log3Stub(Ln ln) {
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
