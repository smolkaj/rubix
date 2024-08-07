* The GUI should start with a random, unsolved cube.
* There should be three buttons:
  1. Scan my cube: Invokes a webcam-based scanner to input real-world cubes.
  2. Shuffle: Randomly shuffles the displayed cube.
  3. Solve: Invokes the solver and visualizes its progress using a progress bar.
     Then shows the solution using the existing GUI.
     * For extra credit, it would be nice to visualize important imermediate
       states of the cube during the solve, to better visualize the solver's
       progress.
* When scanning the cube, the GUI should provide interactive feedback about
  which cubelets are yet unknown and which cubelets have already been
  successfully scanned.
  * The center cubelets are always known.
  * Scanning should infer all cubelets simulataneously instead of scanning faces
    one-by-one. (It may be useful to model the color of each cubelet as a 
    probability distribution over all possible colors. A cubelet is then 
    considered "known" once its distribution is sufficiently concentrated on one
    color, say >= 99%.)
  * In the GUI, perhaps the confidene can be visualized using color intensities.
    For example:
    * Colors below 50% confidence are considered completely unknown and not
      rendered.
    * Between 50-100% confidence, the color is rendered, ranging from pale 
      (low confidence) to vivid (high confidence).
