# Ten Technical Viva Questions — with Accurate Answers

**1. Why use a CNN instead of a plain fully connected network for digit images?**
A CNN reuses small learnable filters across the whole image (weight sharing) and only connects each neuron to a local neighbourhood. This exploits the spatial structure of images, uses far fewer parameters than a dense network on raw pixels, and gives some tolerance to small shifts. A dense network treats every pixel independently and ignores spatial neighbourhoods.

**2. What does each major layer in your network do?**
`Conv2D` layers learn filters that detect edges, strokes and later loops/junctions; ReLU adds non-linearity; `MaxPooling2D` halves the resolution while keeping the strongest responses (shift tolerance, less computation); `Dropout` randomly disables neurons during training to reduce over-fitting; `Flatten` converts feature maps into a vector; `Dense(128)` combines features; `Dense(10, softmax)` outputs ten probabilities summing to 1.

**3. Why softmax in the output layer and sparse categorical cross-entropy as the loss?**
Softmax converts ten raw scores into a probability distribution, which suits single-label multiclass classification. Cross-entropy measures how much probability the model assigned to the true class and penalises confident mistakes. The "sparse" variant accepts integer labels (0–9) directly, so no one-hot encoding is needed; the maths is identical.

**4. Why divide pixel values by 255?**
It rescales 0–255 to 0–1, which keeps inputs in a small, consistent range and makes gradient-based optimisation faster and more stable. The exact same scaling must be applied at inference, which is why the app has one shared preprocessing module.

**5. How did you avoid data leakage between training and testing?**
The 60,000 training images are split (stratified, seeded) into train and validation. Early stopping, learning-rate reduction and the preprocessing stress test use only validation data. The 10,000 test images are used once, after training, to compute the final metrics. No hyper-parameter was chosen using the test set.

**6. What are early stopping and learning-rate reduction, and why use them?**
EarlyStopping monitors validation loss and stops training when it has not improved for a set number of epochs, restoring the best weights; this limits over-fitting. ReduceLROnPlateau halves the learning rate when validation loss stalls, allowing finer weight updates near a minimum.

**7. Explain precision, recall and F1. Why report macro and weighted averages?**
Precision (for a digit) = correct predictions of that digit ÷ all predictions of that digit. Recall = correct predictions ÷ all true instances of that digit. F1 is their harmonic mean. Macro averaging takes the plain mean over the ten classes (each class counts equally); weighted averaging weights each class by its number of test samples. MNIST is nearly balanced, so they are close, but both are reported.

**8. What does a confusion matrix show and how do you read it?**
Rows are true digits, columns predicted digits. Diagonal cells are correct predictions; off-diagonal cells show which digits are confused (for example 4↔9 or 3↔5). Row-normalising gives per-class recall.

**9. Why does an uploaded photo need different preprocessing from MNIST test images?**
MNIST images are already 28×28, grayscale, white-on-black, centred and size-normalised. Real uploads differ in size, colour, background polarity, position, noise and stroke thickness. The pipeline converts to grayscale, detects polarity, removes background/noise, crops, resizes with preserved aspect ratio into a 20×20 box, centres by centre of mass in 28×28 and normalises, so the model sees input that resembles its training distribution.

**10. Does a high softmax confidence mean the prediction is correct? What are the model's limitations?**
No. Softmax probabilities show how the model distributes belief across the ten known classes; they are not calibrated guarantees and can be high on wrong or out-of-distribution inputs (for example a letter). Other limitations: MNIST is clean and limited in writing styles, only single digits are supported, and accuracy on real photos can be lower than the MNIST test accuracy. Improvements include augmentation, calibration, out-of-distribution detection, ensembles and training on more diverse data.
