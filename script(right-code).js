let recent_counts = [];
const zeroPad = (num) => (num < 10 ? "0" : "") + String(num);

const input = document.getElementById("uploadInput");
input.addEventListener("change", async (event) => {
    processImage(event.target.files[0]);
});
/**
 * Function to process the image, apply filters, and trigger object detection
 */
async function processImage(imageFile) {
    document.getElementById("result").innerHTML = `Загрузка (длится ~15 секунд)`;

    // Prepare the image, apply filters if needed
    const { processedImageData, imgWidth, imgHeight } = await prepare_input(imageFile);

    // Run inference on the processed (filtered) image
    const [boxes1, boxes2] = await Promise.all([
        detect_objects_on_image(processedImageData, imgWidth, imgHeight, "src/models/square.onnx"),
        detect_objects_on_image(processedImageData, imgWidth, imgHeight, "src/models/circle.onnx")
    ]);

    // Choose the model with more detected objects
    const boxes = (boxes1.length > boxes2.length ? boxes1 : boxes2);
    const counter = boxes.length;

    // Update the result on the webpage
    const date = new Date();
    document.getElementById("result").innerHTML = `Подсчитано: ${counter}`;
    document.getElementById("recent_counts").style.display = "block";
    
    // Update recent counts and limit to 5 entries
    recent_counts.push([counter, zeroPad(date.getHours()), zeroPad(date.getMinutes())]);
    if (recent_counts.length > 5) recent_counts.shift();  // Keep only 5 results

    // Clear previous results and append new ones
    const infoDiv = document.getElementById("info");
    infoDiv.innerHTML = "";  // Clear previous results
    recent_counts.forEach(([count, hours, minutes]) => {
        const newResult = document.createElement("div");
        newResult.innerHTML = `<p>Объектов: ${count}</p><p>${hours}:${minutes}</p>`;
        infoDiv.prepend(newResult);
    });

    // Draw the filtered image and the bounding boxes
    draw_image_and_boxes(processedImageData, boxes, imgWidth, imgHeight);

    // Reset the input to ensure the same file can be uploaded again with different settings
    document.getElementById("uploadInput").value = null;
}

/**
 * Function to draw the image and the bounding boxes of detected objects
 */
function draw_image_and_boxes(imageData, boxes, imgWidth, imgHeight) {
    const canvas = document.querySelector("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = imgWidth;
    canvas.height = imgHeight;

    // Draw the filtered image
    ctx.putImageData(imageData, 0, 0);

    // Draw bounding boxes
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 3;
    boxes.forEach(([x1, y1, x2, y2]) => {
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    });
}

/**
 * Function to return a new grayscale filtered image data
 */
function apply_grayscale_filter(imageData) {
    const data = imageData.data;
    const newImageData = new ImageData(imageData.width, imageData.height);
    const newPixels = newImageData.data;

    for (let i = 0; i < data.length; i += 4) {
        const grayscale = data[i] * 0.3 + data[i + 1] * 0.59 + data[i + 2] * 0.11;
        newPixels[i] = newPixels[i + 1] = newPixels[i + 2] = grayscale;
        newPixels[i + 3] = data[i + 3];  // Copy alpha channel
    }

    return newImageData;
}

/**
 * Function to return a new sharpen filtered image data
 */
function apply_sharpen_filter(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const pixels = imageData.data;

    const result = new ImageData(width, height);
    const resultPixels = result.data;

    const kernel = [
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    ];

    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const i = (y * width + x) * 4;

            let red = 0, green = 0, blue = 0;

            for (let ky = -1; ky <= 1; ky++) {
                for (let kx = -1; kx <= 1; kx++) {
                    const j = ((y + ky) * width + (x + kx)) * 4;
                    const kernelValue = kernel[(ky + 1) * 3 + (kx + 1)];
                    red += pixels[j] * kernelValue;
                    green += pixels[j + 1] * kernelValue;
                    blue += pixels[j + 2] * kernelValue;
                }
            }

            resultPixels[i] = Math.min(255, Math.max(0, red));
            resultPixels[i + 1] = Math.min(255, Math.max(0, green));
            resultPixels[i + 2] = Math.min(255, Math.max(0, blue));
            resultPixels[i + 3] = pixels[i + 3];  // Keep alpha channel
        }
    }

    return result;
}

function apply_enhance_filter(imageData) {
    const pixels = imageData.data;
    const result = new ImageData(imageData.width, imageData.height);
    const resultPixels = result.data;

    for (let i = 0; i < pixels.length; i += 4) {
        const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];

        const avg = (r + g + b) / 3;
        resultPixels[i] = Math.min(255, r + (avg - r) * 0.2);
        resultPixels[i + 1] = Math.min(255, g + (avg - g) * 0.2);
        resultPixels[i + 2] = Math.min(255, b + (avg - b) * 0.2);
        resultPixels[i + 3] = pixels[i + 3];
    }
    return result;
}

/**
 * Function to prepare input image, resize to 640x640, apply optional filters, and convert it to tensor format
 */
async function prepare_input(file) {
    return new Promise((resolve) => {
        const img = new Image();
        img.src = URL.createObjectURL(file);
        img.onload = () => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");

            // Resize the image to 640x640 to match the model input size
            canvas.width = 640;
            canvas.height = 640;
            ctx.drawImage(img, 0, 0, 640, 640);

            let imageData = ctx.getImageData(0, 0, 640, 640);

            if (document.getElementById('checkbox-filter').checked) {
                imageData = apply_grayscale_filter(imageData);
            }

            if (document.getElementById('checkbox-sharpen').checked) {
                imageData = apply_sharpen_filter(imageData);
            }

            if (document.getElementById('checkbox-enhance').checked) {
                imageData = apply_enhance_filter(imageData);
            }

            // Return the processed image data and dimensions
            resolve({ processedImageData: imageData, imgWidth: 640, imgHeight: 640 });
        };
    });
}

/**
 * Function to pass image through YOLOv8 model and detect objects
 */
async function detect_objects_on_image(imageData, imgWidth, imgHeight, model_name) {
    const input = convertToTensor(imageData, imgWidth, imgHeight);
    const output = await run_model(input, model_name);
    return process_output(output, imgWidth, imgHeight);
}

/**
 * Convert image data into tensor format for the model
 */
function convertToTensor(imageData, imgWidth, imgHeight) {
    const { data: pixels } = imageData;
    const input = Array(640 * 640 * 3).fill(0);
    for (let i = 0, j = 0; i < pixels.length; i += 4, j++) {
        input[j] = pixels[i] / 255.0;          // Red
        input[j + 640 * 640] = pixels[i + 1] / 255.0;  // Green
        input[j + 640 * 640 * 2] = pixels[i + 2] / 255.0;  // Blue
    }
    return input;
}

/**
 * Function to run the object detection model on input
 */
async function run_model(input, model_name) {
    const model = await ort.InferenceSession.create(model_name);
    input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);  // Ensure input is 1x3x640x640
    const outputs = await model.run({ images: input });
    return outputs["output0"].data;
}

/**
 * Process the output of the model and return filtered bounding boxes
 */
function process_output(output, img_width, img_height) {
    const CONFIDENCE_THRESHOLD = document.getElementById("conf-value").value / 100;
    const IOU_THRESHOLD = 0.3;
    let boxes = [];
    console.log(CONFIDENCE_THRESHOLD);
    for (let index = 0; index < 8400; index++) {
        const prob = output[8400 * 4 + index];
        if (prob < CONFIDENCE_THRESHOLD) continue;

        const class_id = getClassIDFromOutput(output, index);
        const [x1, y1, x2, y2] = getBoundingBox(output, index, img_width, img_height);
        boxes.push([x1, y1, x2, y2, class_id]);
    }

    return apply_nms(boxes, IOU_THRESHOLD);
}

/**
 * Get bounding box from output
 */
function getBoundingBox(output, index, img_width, img_height) {
    const xc = output[index];
    const yc = output[8400 + index];
    const w = output[2 * 8400 + index];
    const h = output[3 * 8400 + index];
    const x1 = (xc - w / 2) / 640 * img_width;
    const y1 = (yc - h / 2) / 640 * img_height;
    const x2 = (xc + w / 2) / 640 * img_width;
    const y2 = (yc + h / 2) / 640 * img_height;
    return [x1, y1, x2, y2];
}

/**
 * Get class ID from output
 */
function getClassIDFromOutput(output, index) {
    return [...Array(3).keys()].reduce((accum, col) => {
        const value = output[8400 * (col + 4) + index];
        return value > accum[1] ? [col, value] : accum;
    }, [0, 0])[0];
}

/**
 * Non-Maximum Suppression (NMS) implementation
 */
function apply_nms(boxes, iou_threshold) {
    const result = [];
    while (boxes.length > 0) {
        const chosenBox = boxes.shift();
        result.push(chosenBox);
        boxes = boxes.filter(box => iou(chosenBox, box) < iou_threshold);
    }
    return result;
}

/**
 * Function to calculate Intersection over Union (IoU)
 */
function iou(box1, box2) {
    const [x1_min, y1_min, x1_max, y1_max] = box1;
    const [x2_min, y2_min, x2_max, y2_max] = box2;

    const x_min_intersection = Math.max(x1_min, x2_min);
    const y_min_intersection = Math.max(y1_min, y2_min);
    const x_max_intersection = Math.min(x1_max, x2_max);
    const y_max_intersection = Math.min(y1_max, y2_max);

    const intersection_area = Math.max(0, x_max_intersection - x_min_intersection) *
                              Math.max(0, y_max_intersection - y_min_intersection);

    const box1_area = (x1_max - x1_min) * (y1_max - y1_min);
    const box2_area = (x2_max - x2_min) * (y2_max - y2_min);

    const union_area = box1_area + box2_area - intersection_area;

    return intersection_area / union_area;
}

/**
 * YOLOv8 class labels
 */
const yolo_classes = ['pipes', '0', 'pipe'];
