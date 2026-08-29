package ru.traphouse.functions.trig;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CosTest {
    private static final double EPS = 1e-6;
    private Cos cos;

    @BeforeEach
    void setUp() {
        Sin sin = new Sin(1e-10);
        cos = new Cos(sin);
    }

    @Test
    void testZero() {
        assertEquals(1.0, cos.calculate(0.0), EPS);
    }

    @Test
    void testPiHalf() {
        assertEquals(0.0, cos.calculate(Math.PI / 2), EPS);
    }

    @Test
    void testPi() {
        assertEquals(-1.0, cos.calculate(Math.PI), EPS);
    }

    @Test
    void testPiThird() {
        assertEquals(0.5, cos.calculate(Math.PI / 3), EPS);
    }

    @Test
    void testNegativeValue() {
        assertEquals(Math.cos(-1.0), cos.calculate(-1.0), EPS);
    }

    @Test
    void testMinusTwo() {
        assertEquals(Math.cos(-2.0), cos.calculate(-2.0), EPS);
    }
}
