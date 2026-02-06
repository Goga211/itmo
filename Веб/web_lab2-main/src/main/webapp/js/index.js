class InvalidValueException extends Error {
    constructor(message) {
        super(message);
        this.name = "InvalidValueException";
    }
}

class Validator {
    validate(value) {
        throw new Error("Метод validate() нужно переопределить");
    }
}

class XValidator extends Validator {
    validate(value) {
        if (isNaN(value)) {
            throw new InvalidValueException("Неверное значение X");
        }

        const x = Number(value);
        if (x > 2 || x < -2) {
            throw new InvalidValueException("Число X не входит в диапазон");
        }

        return true;
    }
}

class YValidator extends Validator {
    validate(value) {
        if (isNaN(value)) {
            throw new InvalidValueException("Неверное значение Y");
        }

        const decimalPart = String(value).trim().split('.')[1];
        if (decimalPart && decimalPart.length > 15) {
            throw new InvalidValueException("Слишком много знаков после запятой");
        }

        const x = Number(value);
        if (x < -5 || x > 3) {
            throw new InvalidValueException("Число Y не входит в диапазон");
        }

        return true;
    }
}

class RValidator extends Validator {
    validate(value) {
        if (!value) {
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

// Функция для отправки данных с гибкостью метода
async function submitForm(values, method = 'POST') {
    const formData = new URLSearchParams(values);
    const response = await fetch(`/lab2-1.0-SNAPSHOT/controller-servlet`, {
        method: method,
        headers: {'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'},
        body: formData.toString()
    });

    if (response.ok) {
        const html = await response.text();
        document.open();
        document.write(html);
        document.close();
    } else {
        throw new Error("Ошибка при получении данных с сервера.");
    }
}

// Переменная для хранения последнего значения Y в координатах
let lastYCoord = null;

// Обработчик кликов по SVG
function handleClick(event) {
    const svg = document.getElementById("plate");
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const coords = point.matrixTransform(svg.getScreenCTM().inverse());

    const r = document.querySelector('input[name="r"]:checked')?.value;
    const x = (coords.x - 250) / 20;
    let y = (250 - coords.y) / 20;
    y = Math.round(y);

    // Преобразуем Y обратно в координаты
    const currentYCoord = 250 - (y * 20);

    if (lastYCoord !== null && Math.abs(currentYCoord - lastYCoord) <= 5) {
        const values = {
            x: x.toFixed(2),
            y: String(y),
            r: String(r)
        };

        try {
            validateFormInput(values);
            submitForm(values).catch(error => alert(error.message));
        } catch (e) {
            alert(e.message);
        }
    } else {
        lastYCoord = currentYCoord; // Обновляем последнее значение Y в координатах
    }
}

// Универсальная обработка формы
async function handleSubmitForm(ev, form) {
    ev.preventDefault();
    const formData = new FormData(form);
    const values = {
        x: formData.get('x'),
        y: formData.get('y'),
        r: formData.get('r')
    };

    const errorDiv = document.getElementById("error");

    try {
        validateFormInput(values);
        errorDiv.hidden = true;
        await submitForm(values);
    } catch (e) {
        errorDiv.hidden = false;
        errorDiv.textContent = e.message;
    }
}

document.addEventListener("DOMContentLoaded", function () {
    document.getElementById("data-form").addEventListener('submit', (ev) => handleSubmitForm(ev, ev.target));
    document.getElementById("plate").addEventListener("click", handleClick);
});

document.addEventListener("DOMContentLoaded", function () {
    // Центр и базовая шкала: при r=5 → Rpx=100
    const CX = 250, CY = 250;
    const r_px = 100; // пикселей на R при r=5

    const rLabels = Array.from(document.querySelectorAll('text.small'));
    rLabels.forEach(lbl => {
        const x = parseFloat(lbl.getAttribute('x') || '0');
        const y = parseFloat(lbl.getAttribute('y') || '0');
        let role;
        if (Math.abs(y - CY) <= 12) {              // метки на оси X
            role = (x < CX) ? 'label-neg-rx' : 'label-rx';
        } else {                                   // метки на оси Y
            role = (y > CY) ? 'label-neg-ry' : 'label-ry';
        }
        lbl.dataset.role = role;
    });

    let radiusEl = document.querySelector('input[name="r"]:checked');
    let radius = radiusEl ? parseFloat(radiusEl.value) : 5;
    updateGraph(radius);

    const radioButtons = document.querySelectorAll('input[name="r"]');
    radioButtons.forEach(radio => {
        radio.addEventListener('change', function () {
            radius = parseFloat(this.value);
            updateGraph(radius);
        });
    });

    function updateGraph(r) {
        const scaleFactor = Number(r) / 5;
        const len_r  = r_px * scaleFactor; // длина R в пикселях
        const R2px = len_r / 2;                 // R/2 в пикселях

        const rect = document.getElementById("rect");
        rect.setAttribute("x", CX - len_r);
        rect.setAttribute("y", CY);
        rect.setAttribute("width",  len_r);
        rect.setAttribute("height", len_r);

        // --- 2-я четверть: прямоугольный треугольник с катетами R/2
        const tri = document.getElementById("triangle");
        tri.setAttribute("points",`${CX},${CY} ${CX - R2px},${CY} ${CX},${CY - R2px}`);

        // --- 1-я четверть: четверть круга радиуса R/2
        const arc = document.getElementById("arc");
        arc.setAttribute("d", `M ${CX},${CY} ` + `L ${CX + R2px},${CY} ` + `A ${R2px} ${R2px} 0 0 0 ${CX} ${CY - R2px} ` + `Z`);

    }
});

