import React, { useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import styled from "styled-components";
import {
    incrementScaleCounter,
    triggerRefresh,
} from "../../redux/generalSlice";
import axiosInstance from "../../axiosInstance";

const Svg = styled.svg`
    margin-top: 3%;
    width: 100%;
    height: 500px;
`;

const Line = styled.line`
    stroke: silver;
    stroke-width: 2;
`;

const Polygon = styled.polygon`
    fill: white;
`;

const Rect = styled.rect`
    fill: white;
`;

const Path = styled.path`
    fill: white;
`;

const Text = styled.text`
    fill: white;
    font-size: 14px;
`;

const ErrorMessage = styled.div`
    color: red;
    font-size: 14px;
    margin-top: 10px;
    text-align: center;
`;

// 1 координатная единица = 33 px
const SCALE = 33;

function CoordinatePlate() {
    const table = useSelector((state) => state.tableEditor.table);
    const radius = useSelector((state) => state.tableEditor.radius);
    const loading = useSelector((state) => state.tableEditor.loading);

    const dispatch = useDispatch();
    const beginPlate = 250;

    const [error, setError] = useState("");

    const rawR = Number(radius) || 0;
    const absR = Math.abs(rawR);
    const sign = rawR >= 0 ? 1 : -1;

    /* ===============================
        ✅ КЛИК (обратный отрисовке)
       =============================== */
    const handlePlateClick = async (event) => {
        if (absR === 0) return;

        const svg = event.currentTarget;
        const point = svg.createSVGPoint();

        point.x = event.clientX;
        point.y = event.clientY;

        const tp = point.matrixTransform(svg.getScreenCTM().inverse());

        // 1 ед. = 33 px, только sign
        const x = ((tp.x - beginPlate) / SCALE) * sign;
        const y = ((beginPlate - tp.y) / SCALE) * sign;

        const values = {
            x: +x.toFixed(2),
            y: +y.toFixed(2),
            r: rawR,
        };

        try {
            await axiosInstance.post("/dot/check", values);
            dispatch(incrementScaleCounter());
            dispatch(triggerRefresh());
        } catch (err) {
            console.error("Error sending dot:", err);
            setError(
                err?.response?.data?.message ||
                err?.response?.data?.error ||
                "Request failed"
            );
        }
    };

    /* ===============================
        ✅ ТОЧКИ: зависят и от dot.r, и от текущего rawR
       =============================== */
    const dots = loading
        ? []
        : table.map((dot, index) => {
            if (absR === 0 || dot.r === 0) return null;

            // ✅ коэффициент расширения относительно текущего maxR
            const k = absR / Math.abs(dot.r);

            const xScaled = dot.x * k;
            const yScaled = dot.y * k;

            const xPx = beginPlate + xScaled * SCALE * sign;
            const yPx = beginPlate - yScaled * SCALE * sign;

            return (
                <circle
                    key={index}
                    cx={xPx}
                    cy={yPx}
                    r="2"
                    fill={dot.result ? "green" : "red"}   // ✅ ТОЛЬКО С БЕКА
                />
            );
        });


    /* ===============================
        ✅ ФИГУРЫ (по |R| * SCALE)
       =============================== */

    // --- ТРЕУГОЛЬНИК ---
    let trianglePoints = "";
    if (absR > 0) {
        const d = (absR / 2) * SCALE;

        if (rawR >= 0) {
            trianglePoints = `
                ${beginPlate - d}, ${beginPlate}
                ${beginPlate}, ${beginPlate - d}
                ${beginPlate}, ${beginPlate}
            `;
        } else {
            trianglePoints = `
                ${beginPlate + d}, ${beginPlate}
                ${beginPlate}, ${beginPlate + d}
                ${beginPlate}, ${beginPlate}
            `;
        }
    }

    // --- ПРЯМОУГОЛЬНИК ---
    let rectX = 0, rectY = 0, rectW = 0, rectH = 0;
    if (absR > 0) {
        rectW = absR * SCALE;
        rectH = (absR / 2) * SCALE;

        if (rawR >= 0) {
            rectX = beginPlate - rectW;
            rectY = beginPlate;
        } else {
            rectX = beginPlate;
            rectY = beginPlate - rectH;
        }
    }

    // --- ЧЕТВЕРТЬ КРУГА ---
    let sectorPath = "";
    if (absR > 0) {
        const rpx = (absR / 2) * SCALE;

        if (rawR >= 0) {
            sectorPath = `
                M ${beginPlate} ${beginPlate}
                h ${rpx}
                a ${rpx} ${rpx} 0 0 1 ${-rpx} ${rpx}
                Z
            `;
        } else {
            sectorPath = `
                M ${beginPlate} ${beginPlate}
                h ${-rpx}
                a ${rpx} ${rpx} 0 0 1 ${rpx} ${-rpx}
                Z
            `;
        }
    }

    return (
        <>
            <Svg viewBox="0 0 500 500" onClick={handlePlateClick}>
                {/* ОСИ */}
                <Line x1="50" y1={beginPlate} x2="450" y2={beginPlate} />
                <Line x1={beginPlate} y1="50" x2={beginPlate} y2="450" />

                {/* МЕТКИ R */}
                {absR > 0 && (
                    <>
                        {/* X: +R/2 и +R */}
                        <Text x={beginPlate + (absR / 2) * SCALE} y={beginPlate + 20}>R/2</Text>
                        <Text x={beginPlate + absR * SCALE} y={beginPlate + 20}>R</Text>

                        {/* X: -R/2 и -R */}
                        <Text x={beginPlate - (absR / 2) * SCALE} y={beginPlate + 20}>-R/2</Text>
                        <Text x={beginPlate - absR * SCALE} y={beginPlate + 20}>-R</Text>

                        {/* Y: +R/2 и +R */}
                        <Text x={beginPlate + 10} y={beginPlate - (absR / 2) * SCALE}>R/2</Text>
                        <Text x={beginPlate + 10} y={beginPlate - absR * SCALE}>R</Text>

                        {/* Y: -R/2 и -R */}
                        <Text x={beginPlate + 10} y={beginPlate + (absR / 2) * SCALE}>-R/2</Text>
                        <Text x={beginPlate + 10} y={beginPlate + absR * SCALE}>-R</Text>
                    </>
                )}

                {/* СТРЕЛКИ */}
                <Polygon points="450,245 450,255 460,250" />
                <Polygon points="245,50 255,50 250,40" />

                {/* ФИГУРЫ */}
                {absR > 0 && (
                    <>
                        <Polygon points={trianglePoints} />
                        <Rect
                            x={rectX}
                            y={rectY}
                            width={rectW}
                            height={rectH}
                        />
                        <Path d={sectorPath} />
                    </>
                )}

                {/* ПОДПИСИ ОСЕЙ */}
                <Text x="450" y="240">X</Text>
                <Text x="260" y="50">Y</Text>

                {/* ТОЧКИ */}
                {dots}
            </Svg>

            {error && <ErrorMessage>{error}</ErrorMessage>}
        </>
    );
}

export default CoordinatePlate;
