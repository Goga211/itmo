import React, { useState } from "react";
import { useFormik } from "formik";
import * as Yup from "yup";
import { Button } from "primereact/button";
import { Checkbox } from "primereact/checkbox";
import { Slider } from "primereact/slider";
import { InputNumber } from "primereact/inputnumber";
import { useDispatch } from "react-redux";
import { triggerRefresh, updateGraph } from "../../redux/generalSlice";
import axiosInstance from "../../axiosInstance";
import styled from "styled-components";

const FormContainer = styled.form`
    margin-top: 20px;
    width: 280px;
    display: flex;
    flex-direction: column;
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
`;

const Section = styled.div`
    margin-bottom: 20px;
`;

const CheckboxGrid = styled.div`
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
`;

const ErrorText = styled.div`
    color: red;
    margin-bottom: 10px;
    font-size: 14px;
`;

const VALUES = [-3, -2, -1, 0, 1, 2, 3, 4, 5];

function CoordinateForm() {
    const [error, setError] = useState("");
    const dispatch = useDispatch();

    const formik = useFormik({
        initialValues: {
            x: [],
            y: 0,
            r: [],
        },

        validationSchema: Yup.object({
            x: Yup.array().min(1, "Выбери хотя бы один X"),
            y: Yup.number()
                .required()
                .min(-3, "Y ≥ -3")
                .max(5, "Y ≤ 5"),
            r: Yup.array()
                .min(1, "Выбери хотя бы один R")
                .notOneOf([[0]], "R не может быть 0"),
        }),

        onSubmit: async (values, { setSubmitting }) => {
            setError("");

            const requests = [];

            for (const x of values.x) {
                for (const r of values.r) {
                    requests.push(
                        axiosInstance.post("/dot/check", {
                            x,
                            y: values.y,
                            r,
                        })
                    );
                }
            }

            try {
                await Promise.all(requests);
                dispatch(triggerRefresh());
            } catch (e) {
                console.error(e);
                setError("Ошибка отправки");
            } finally {
                setSubmitting(false);
            }
        },
    });

    const toggleValue = (field, value) => {
        const current = formik.values[field];
        const updated = current.includes(value)
            ? current.filter((v) => v !== value)
            : [...current, value];

        formik.setFieldValue(field, updated);

        if (field === "r" && updated.length > 0) {
            const maxAbs = Math.max(...updated.map(v => Math.abs(v)));
            const signedMaxR = updated.find(v => Math.abs(v) === maxAbs);
            dispatch(updateGraph(signedMaxR));
        }
    };

    return (
        <FormContainer onSubmit={formik.handleSubmit}>

            {formik.errors.x && <ErrorText>{formik.errors.x}</ErrorText>}
            {formik.errors.y && <ErrorText>{formik.errors.y}</ErrorText>}
            {formik.errors.r && <ErrorText>{formik.errors.r}</ErrorText>}
            {error && <ErrorText>{error}</ErrorText>}

            {/* ✅ X */}
            <Section>
                <div className="mb-2 font-bold">X coordinate</div>
                <CheckboxGrid>
                    {VALUES.map((val) => (
                        <div key={val} className="flex align-items-center">
                            <Checkbox
                                inputId={`x-${val}`}
                                checked={formik.values.x.includes(val)}
                                onChange={() => toggleValue("x", val)}
                            />
                            <label htmlFor={`x-${val}`} className="ml-2">
                                {val}
                            </label>
                        </div>
                    ))}
                </CheckboxGrid>
            </Section>

            {/* ✅ Y */}
            <Section>
                <div className="mb-2 font-bold">Y coordinate: {formik.values.y}</div>
                <Slider
                    value={formik.values.y}
                    min={-3}
                    max={5}
                    step={0.01}
                    onChange={(e) => formik.setFieldValue("y", e.value)}
                />
                <InputNumber
                    className="mt-2 w-full"
                    value={formik.values.y}
                    min={-3}
                    max={5}
                    step={0.01}
                    onValueChange={(e) => formik.setFieldValue("y", e.value)}
                />
            </Section>

            {/* ✅ R */}
            <Section>
                <div className="mb-2 font-bold">Radius</div>
                <CheckboxGrid>
                    {VALUES.map((val) => (
                        <div key={val} className="flex align-items-center">
                            <Checkbox
                                inputId={`r-${val}`}
                                checked={formik.values.r.includes(val)}
                                onChange={() => toggleValue("r", val)}
                            />
                            <label htmlFor={`r-${val}`} className="ml-2">
                                {val}
                            </label>
                        </div>
                    ))}
                </CheckboxGrid>
            </Section>

            <Button
                label="Submit"
                type="submit"
                loading={formik.isSubmitting}
                disabled={formik.isSubmitting}
                className="w-full"
            />
        </FormContainer>
    );
}

export default CoordinateForm;
