package ru.traphouse.stubs.log;

import ru.traphouse.functions.log.Ln;

import java.util.HashMap;
import java.util.Map;

public class LnStub extends Ln {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(1.0,    0.0);
        TABLE.put(2.0,    0.6931471805599453);
        TABLE.put(3.0,    1.0986122886681098);
        TABLE.put(5.0,    1.6094379124341003);
        TABLE.put(10.0,   2.302585092994046);
        TABLE.put(0.5,   -0.6931471805599453);
        TABLE.put(0.1,   -2.302585092994046);
        TABLE.put(Math.E, 1.0);
    }

    public LnStub() {
        super(1e-10);
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
