package ru.traphouse.stubs.trig;

import ru.traphouse.functions.trig.Cos;
import ru.traphouse.functions.trig.Sin;

import java.util.HashMap;
import java.util.Map;

public class CosStub extends Cos {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(0.0,           1.0);
        TABLE.put(Math.PI / 3,   0.5);
        TABLE.put(Math.PI / 2,   0.0);
        TABLE.put(Math.PI,      -1.0);
        TABLE.put(-1.0,          0.5403023058681398);
        TABLE.put(-2.0,         -0.4161468365471424);
        TABLE.put(-3.0,         -0.9899924966004454);
    }

    public CosStub(Sin sin) {
        super(sin);
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
