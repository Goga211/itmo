package ru.traphouse.functions.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CotTest {
    private static final double EPS = 1e-6;
    private Cot cot;

    @BeforeEach
    void setUp() {
        Sin sin = new Sin(1e-10);
        Cos cos = new Cos(sin);
        cot = new Cot(sin, cos);
    }

    @Test
    void testPiQuarter() {
        assertEquals(1.0, cot.calculate(Math.PI / 4), EPS);
    }

    @Test
    void testMinusPiQuarter() {
        assertEquals(-1.0, cot.calculate(-Math.PI / 4), EPS);
    }

    @Test
    void testKnownValue() {
        assertEquals(Math.cos(-1.0) / Math.sin(-1.0), cot.calculate(-1.0), EPS);
    }

    @Test
    void testUndefinedAtZero() {
        assertThrows(ArithmeticException.class, () -> cot.calculate(0.0));
    }

    @Test
    void testUndefinedAtPi() {
        assertThrows(ArithmeticException.class, () -> cot.calculate(Math.PI));
    }
}
