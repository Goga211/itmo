// main.js
document.addEventListener("DOMContentLoaded", function () {
    const center_coordinate_plate = 250;

    let mutation_counter = 0;
    let counter = 0;

    const table = document.getElementById("result-table");
    const rInput = document.querySelector('input[id$=":rInput"]');
    const rDisplayEl = document.getElementById("data-form:rDisplay");

    // --- вспомогалка: текущий R из hidden ---
    function getCurrentR() {
        if (!rInput || !rInput.value) {
            return NaN;
        }
        // на сервер мы шлём формат с точкой (en_US),
        // но на всякий случай убираем запятую
        const raw = rInput.value.replace(",", ".");
        return parseFloat(raw);
    }

    function isTableEmpty() {
        if (!table) return true;
        const rows = table.querySelectorAll("tr:not(:first-child)");
        return rows.length === 0;
    }

    // ====== ЧАСОВОЙ ПОЯС ======
    function changeTimeZone() {
        const clientTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        document.querySelectorAll(".now-time").forEach(cell => {
            const mscTimeString = cell.textContent.trim();
            if (!mscTimeString) return;

            const parts = mscTimeString.split(" ");
            if (parts.length < 2) return;

            const [datePart, timePart] = parts;
            const moscowDate = new Date(`${datePart}T${timePart}+03:00`);

            if (!isNaN(moscowDate)) {
                const options = {
                    timeZone: clientTimeZone,
                    year: "numeric",
                    month: "2-digit",
                    day: "2-digit",
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                    fractionalSecondDigits: 3,
                };

                const clientTime = new Intl.DateTimeFormat("en-GB", options).format(moscowDate);
                cell.textContent = clientTime.replaceAll("/", "-").replace(",", "");
            }
        });
    }

    // ====== ВАЛИДАЦИЯ ======
    class InvalidValueException extends Error {
        constructor(message) {
            super(message);
            this.name = "InvalidValueException";
        }
    }

    class Validator {
        validate(_) {
            throw new Error("Метод validate() нужно переопределить");
        }
    }

    class YValidator extends Validator {
        validate(value) {
            if (isNaN(value)) {
                throw new InvalidValueException("Неверное значение Y");
            }

            const decimalPart = String(value).trim().split(".")[1];
            if (decimalPart && decimalPart.length > 15) {
                throw new InvalidValueException("Слишком много знаков после запятой");
            }

            const y = Number(value);
            if (y < -3 || y > 3) {
                throw new InvalidValueException("Число Y не входит в диапазон");
            }

            return true;
        }
    }

    class XValidator extends Validator {
        validate(value) {
            if (isNaN(value)) {
                throw new InvalidValueException("Неверное значение X");
            }

            const x = Number(value);
            if (x < -2 || x > 2) {
                throw new InvalidValueException("Число X не входит в диапазон");
            }
            return true;
        }
    }

    class RValidator extends Validator {
        validate(value) {
            if (value === null || value === undefined || value === "" || isNaN(value)) {
                throw new InvalidValueException("Пожалуйста, выберите значение R");
            }
            return true;
        }
    }

    const xValidator = new XValidator();
    const yValidator = new YValidator();
    const rValidator = new RValidator();

    function validateFormInput(values) {
        xValidator.validate(values.x);
        yValidator.validate(values.y);
        rValidator.validate(values.r);
    }

    // ====== ВЫБОР X (кнопки) ======
    window.selectX = function (btn) {
        const value = btn.getAttribute("data-x");

        const xInput = document.querySelector('input[id$=":x"]');
        if (xInput) {
            xInput.value = value;
        }

        document.querySelectorAll(".x-button").forEach(b => {
            b.classList.remove("x-button--active");
        });

        btn.classList.add("x-button--active");

        const errorDiv = document.getElementById("error");
        if (errorDiv) {
            errorDiv.hidden = true;
        }
    };

    (function initXButtons() {
        const xInput = document.querySelector('input[id$=":x"]');
        if (!xInput || !xInput.value) return;

        document.querySelectorAll(".x-button").forEach(b => {
            if (b.getAttribute("data-x") === xInput.value) {
                b.classList.add("x-button--active");
            }
        });
    })();

    // ====== КЛИК ПО SVG ======
    function handleClick(event) {
        const svg = document.getElementById("plate");
        if (!svg) return;

        const point = svg.createSVGPoint();
        point.x = event.clientX;
        point.y = event.clientY;
        const coords = point.matrixTransform(svg.getScreenCTM().inverse());

        const rVal = getCurrentR();

        let x = (coords.x - 250) / 33;
        let y = (250 - coords.y) / 33;

        try {
            validateFormInput({ x: x.toFixed(2), y: y.toFixed(2), r: rVal });

            const xInput = document.querySelector('input[id$=":x"]');
            const yInput = document.querySelector('input[id$=":y"]');

            if (xInput) {
                xInput.value = x.toFixed(2);
            }
            if (yInput) {
                yInput.value = y.toFixed(2);
            }

            const submitBtn = document.getElementById("data-form:submit");
            if (submitBtn) {
                submitBtn.click();
            }
        } catch (e) {
            alert(e.message);
        }
    }

    let allowAjaxRequest = false;

    window.validateAndSubmitForm = function () {
        const form = document.getElementById("data-form");
        const formData = new FormData(form);

        const rValue = getCurrentR();

        const xInput = document.querySelector('input[id$=":x"]');
        const xValue = xInput ? xInput.value : null;

        const values = {
            x: xValue || formData.get("data-form:x"),
            y: formData.get("data-form:y"),
            r: rValue
        };

        const errorDiv = document.getElementById("error");

        try {
            validateFormInput(values);
            errorDiv.hidden = true;
            allowAjaxRequest = true;
            return true;
        } catch (e) {
            errorDiv.hidden = false;
            errorDiv.textContent = e.message;
            allowAjaxRequest = false;
            return false;
        }
    };

    const plate = document.getElementById("plate");
    if (plate) {
        plate.addEventListener("click", handleClick);
    }

    // ====== ГРАФИК (ГЛОБАЛЬНЫЕ ФУНКЦИИ!) ======
    window.updateGraph = function (r) {
        if (isNaN(r)) return;

        const scaleFactor = r / 3;

        document.getElementById("rect").setAttribute("width", 99 * scaleFactor);
        document.getElementById("rect").setAttribute("height", 100 * scaleFactor);
        document.getElementById("rect").setAttribute("x", 250 - 100 * scaleFactor);
        document.getElementById("rect").setAttribute("y", 251);

        document
            .getElementById("arc")
            .setAttribute(
                "d",
                `M ${250 - 100 * scaleFactor} 250 A ${100 * scaleFactor} ${100 * scaleFactor} 0 0 1 250 ${
                    250 - 100 * scaleFactor
                } L 250 250 Z`
            );

        document
            .getElementById("triangle")
            .setAttribute(
                "points",
                `251,249 251,${250 - 50 * scaleFactor} ${250 + 50 * scaleFactor},249`
            );

        document.getElementById("mark-neg-rx").setAttribute("x1", 250 - 100 * scaleFactor);
        document.getElementById("mark-neg-rx").setAttribute("x2", 250 - 100 * scaleFactor);

        document.getElementById("mark-rx").setAttribute("x1", 250 + 100 * scaleFactor);
        document.getElementById("mark-rx").setAttribute("x2", 250 + 100 * scaleFactor);

        document.getElementById("mark-ry").setAttribute("y1", 250 - 100 * scaleFactor);
        document.getElementById("mark-ry").setAttribute("y2", 250 - 100 * scaleFactor);

        document.getElementById("mark-neg-ry").setAttribute("y1", 250 + 100 * scaleFactor);
        document.getElementById("mark-neg-ry").setAttribute("y2", 250 + 100 * scaleFactor);

        document.getElementById("label-neg-rx").setAttribute("x", 250 - 120 * scaleFactor);
        document.getElementById("label-rx").setAttribute("x", 250 + 103 * scaleFactor);

        document.getElementById("label-neg-ry").setAttribute("y", 250 + 110 * scaleFactor);
        document.getElementById("label-ry").setAttribute("y", 250 - 96 * scaleFactor);

        // перерисовка точек под новым масштабом
        window.drawPoints(scaleFactor, false);
    };

    window.drawPoints = function (scale = 1, new_point = false) {
        console.log("drawPoints called with r = ", scale, "and new_point = ", new_point);

        let svg = document.getElementById("plate");
        if (!svg) return;

        svg.querySelectorAll(".data-point").forEach(point => point.remove());

        let points = document.querySelectorAll("#points-data .point");

        let pointsArray;
        let otherPoints;
        let lastPoints;

        if (new_point === true) {
            counter++;
            pointsArray = Array.from(points);
            lastPoints = pointsArray.slice(-counter);
            otherPoints = pointsArray.slice(0, pointsArray.length - counter);
        } else {
            lastPoints = 0;
            otherPoints = Array.from(points);
            counter = 0;
        }

        otherPoints.forEach(point => {
            const x = parseFloat(point.getAttribute("data-x"));
            const y = parseFloat(point.getAttribute("data-y"));
            const result = point.getAttribute("data-result") === "true";

            const svgX = center_coordinate_plate + x * (scale * 33);
            const svgY = center_coordinate_plate - y * (scale * 33);

            const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
            circle.setAttribute("cx", svgX);
            circle.setAttribute("cy", svgY);
            circle.setAttribute("r", 2);
            circle.setAttribute("fill", result ? "green" : "red");
            circle.classList.add("data-point");

            svg.appendChild(circle);
        });

        if (lastPoints !== 0) {
            lastPoints.forEach(point => {
                const x = parseFloat(point.getAttribute("data-x"));
                const y = parseFloat(point.getAttribute("data-y"));
                const result = point.getAttribute("data-result") === "true";

                const svgX = center_coordinate_plate + x * 33;
                const svgY = center_coordinate_plate - y * 33;

                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", svgX);
                circle.setAttribute("cy", svgY);
                circle.setAttribute("r", 2);
                circle.setAttribute("fill", result ? "green" : "red");
                circle.classList.add("data-point");

                svg.appendChild(circle);
            });
        }
    };

    // ====== MutationObserver для таблицы результатов ======
    let debounceTimer = null;

    const resultElement = document.getElementById("result");
    if (resultElement) {
        const observer = new MutationObserver(function () {
            observer.disconnect();

            if (debounceTimer) {
                clearTimeout(debounceTimer);
            }

            debounceTimer = setTimeout(() => {
                mutation_counter++;
                changeTimeZone();

                if (mutation_counter !== 1 || isTableEmpty()) {
                    const rNum = getCurrentR();
                    if (!isNaN(rNum)) {
                        window.drawPoints(rNum / 3, true);
                    }
                }

                debounceTimer = null;

                observer.observe(resultElement, {
                    childList: true,
                    subtree: true
                });
            }, 100);
        });

        observer.observe(resultElement, {
            childList: true,
            subtree: true
        });
    }

    // ====== Старт: рисуем график по начальному R, если он есть ======
    const initR = getCurrentR();
    if (!isNaN(initR)) {
        window.updateGraph(initR);
        if (rDisplayEl) {
            rDisplayEl.textContent = initR.toString();
        }
    }

    changeTimeZone();
});
