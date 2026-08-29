package ru.traphouse.system;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ru.traphouse.functions.trig.*;
import ru.traphouse.stubs.trig.*;

import static org.junit.jupiter.api.Assertions.*;

class TrigSystemTest {
    private static final double EPS = 1e-5;
    private TrigSystem trigSystem;
    private TrigSystem trigSystemWithStubs;

    @BeforeEach
    void setUp() {
        Sin sin = new Sin(1e-10);
        Cos cos = new Cos(sin);
        Cot cot = new Cot(sin, cos);
        Csc csc = new Csc(sin);
        trigSystem = new TrigSystem(sin, cot, csc);

        SinStub sinStub = new SinStub();
        CosStub cosStub = new CosStub(sinStub);
        CotStub cotStub = new CotStub(sinStub, cosStub);
        CscStub cscStub = new CscStub(sinStub);
        trigSystemWithStubs = new TrigSystem(sinStub, cotStub, cscStub);
    }

    @Test
    void testUndefinedAtZero() {
        assertThrows(ArithmeticException.class, () -> trigSystem.calculate(0.0));
    }

    @Test
    void testUndefinedAtMinusPi() {
        assertThrows(ArithmeticException.class, () -> trigSystem.calculate(-Math.PI));
    }

    @Test
    void testMinusOne() {
        double x = -1.0;
        double expected = Math.pow((Math.cos(x) - 1.0) / Math.sin(x), 6);
        assertEquals(expected, trigSystem.calculate(x), EPS);
    }

    @Test
    void testMinusTwo() {
        double x = -2.0;
        double expected = Math.pow((Math.cos(x) - 1.0) / Math.sin(x), 6);
        assertEquals(expected, trigSystem.calculate(x), EPS);
    }

    @Test
    void testStubMatchesReal() {
        double x = -1.0;
        // Возведение в 6-ю степень усиливает разницу между stub (Math.sin) и серией,
        // поэтому допуск здесь выше, чем для базовых функций.
        assertEquals(trigSystem.calculate(x), trigSystemWithStubs.calculate(x), 1e-4);
    }

    @Test
    void testMinusPiHalf() {
        // sin(-π/2) = -1, cos(-π/2) = 0 → inner = (0-1) = -1 → (-1)^6 = 1
        assertEquals(1.0, trigSystem.calculate(-Math.PI / 2), EPS);
    }
}
