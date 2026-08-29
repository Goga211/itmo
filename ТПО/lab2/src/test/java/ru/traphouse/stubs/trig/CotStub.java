package ru.traphouse.stubs.trig;

import ru.traphouse.functions.trig.Cos;
import ru.traphouse.functions.trig.Cot;
import ru.traphouse.functions.trig.Sin;

import java.util.HashMap;
import java.util.Map;

public class CotStub extends Cot {
    private static final Map<Double, Double> TABLE = new HashMap<>();

    static {
        TABLE.put(Math.PI / 4,   1.0);
        TABLE.put(Math.PI / 2,   0.0);
        TABLE.put(-Math.PI / 4, -1.0);
        TABLE.put(-1.0,         -0.6421289781069243);
        TABLE.put(-2.0,          0.4576575543182522);
        TABLE.put(-3.0,          7.015252551434534);
    }

    public CotStub(Sin sin, Cos cos) {
        super(sin, cos);
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
