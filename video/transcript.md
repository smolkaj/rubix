# A cube made of transformations

## 00:00:00 — Teach a computer one turn

Imagine you are writing a program for a Rubik's cube. Before it can find a solution, it needs to do something much more basic. Turn the top layer correctly. Watch. The white face turns, but strips of stickers on the sides move too. Which colors go where? You could record each sticker and write rules for moving it. That works. But the real cube never consults a list. Its shape tells the colors how to move. Can we give our program a description that makes those same rules follow from the geometry? That is the problem we will solve.

## 00:00:45 — First, what are we moving?

Start with one small colored square. We will call it a sticker, or facelet. Nine stickers make up one large face. There are six faces, and fifty four stickers in all. A face is an outside surface. The objects that carry those stickers live underneath it. Let's look at one.

## 00:01:09 — A list can record the colors

Here is a natural first attempt. Number the fifty four sticker positions, and store their colors in a list. This is a perfectly valid way to represent the cube. Now turn the top. Each moving sticker has to jump from its old list entry to a new one. Follow the colored paths. A single smooth turn has become a collection of scattered index changes. We could specify those permutations, one move at a time. But the list order gives us little help in seeing why these stickers belong to the same turn. Can we keep the spatial structure in our description?

## 00:01:55 — Two stickers. One rigid cubelet.

This little block is a cubelet. It sits on an edge, between two large faces. That is why it has two stickers: green and white. The stickers are attached to the same rigid object. However we turn it, they stay together, at a right angle. Our description should preserve that relationship automatically.

## 00:02:21 — A turn moves a whole layer

Put that cubelet back. When we say turn the top face, we physically rotate the layer of nine cubelets behind it. This layer is also called an outer slice. Watch the highlighted layer make a quarter turn. The other eighteen positions stay put. The colored squares on the sides move because their cubelets move. So we need two rules. Choose a layer. Then rotate everything in it together.

## 00:02:56 — A 3D array keeps the shape

The cube suggests a better starting point for our purpose. Arrange its small blocks in a three by three by three array. One slot per cubelet position, with its stickers attached. Now the top layer is an entire plane of this grid. That makes it easier to see what belongs together. But we still need a rule for moving those blocks and changing which way their stickers face. Is there a simple language that describes a turn of this three dimensional array? A language where the motion of the colors follows from the motion of the blocks?

## 00:03:38 — We need location and facing

Follow our edge through that turn. Where it sits changes. Which way its green sticker points changes too. Knowing only its position would not tell us where to draw its colors. We need both location and facing. And because the stickers are rigidly attached, the same physical turn changes both. Could one mathematical operation do both jobs?

## 00:04:07 — Build the description, one idea at a time

First we will give each cubelet an address. Then we will see what that address tells us about its stickers. Finally, we will put the description into motion. No matrix is needed yet. First, let's make the geometry feel familiar.

## 00:04:26 — Find something that stays put

We need a way to describe where things are. But what can we use as a reference while the cubelets move? Look at the six centers. Turn the top layer again. Its center spins in place. The other centers stay where they were. Outer-layer turns never exchange these six positions. Imagine a cross joining the opposite centers through the middle of the puzzle. Those fixed directions give us a frame for every move. We keep the whole cube in this frame, instead of turning it in our hands.

## 00:05:04 — Three directions, measured from the middle

Put zero at the middle of the puzzle. We will call the direction toward green x, toward red y, and toward white z. One step toward green is positive x. One step the other way, toward blue, is negative x. Zero means we have not moved along that direction. The other two axes work the same way. These names are a choice. What matters is using the same fixed directions throughout.

## 00:05:37 — Read an address as three instructions

Let's locate our edge from the middle. First, take one step toward green. That is the x coordinate: one. Next, take zero steps toward red. The y coordinate is zero. Finally, take one step up toward white. The z coordinate is one. One, zero, one. Three numbers, always in x, y, z order. Together they name exactly this position. You can read a coordinate as a little set of walking instructions.

## 00:06:17 — The whole cube fits on a small grid

Each direction has only three levels: minus one, zero, and one. The same three numbers locate every block in a three by three by three grid. That gives twenty seven positions. This is an idealized geometric model of the puzzle, not a diagram of its internal mechanism. The middle position has no stickers. Rubix stores the other twenty six.

## 00:06:47 — An address can also be an arrow

Now draw one arrow from the middle to our edge. Its sideways and upward components are exactly those walking instructions. This is a vector: an arrow described by three numbers. We can write the numbers vertically as a column. The first row is x, the second is y, the third is z. It means the same thing as one, zero, one written across the page. Why draw an arrow? Because an arrow can turn with the cube. Soon that will let the geometry do our bookkeeping.

## 00:07:26 — Give the cubelet a permanent address

This address describes where the green and white edge belongs in the solved cube. Call that home vector c. Keep c fixed, even after a scramble. It names the same cubelet throughout the story. Later we will use a different vector for where that cubelet is now.

## 00:07:48 — The boundary tells us how many stickers

Look at a center cubelet. Its home address has one nonzero coordinate. It touches one outside boundary, and carries one sticker. An edge has two nonzero coordinates. Two outside boundaries. Two stickers. A corner has three: three boundaries, three stickers. This is our first payoff. A coordinate of zero stays in the middle along that axis. A coordinate of plus or minus one reaches an outside surface. Counting the nonzero coordinates counts the stickers!

## 00:08:29 — Signs tell sides. Magnitudes count boundaries.

For our edge, one, zero, one gives one plus zero plus one. Two stickers. Now consider the opposite edge, minus one, zero, minus one. It still has two stickers. The minus signs choose the opposite faces; they do not subtract stickers. Take absolute values, then add. That little formula counts the exposed surfaces of every cubelet. Our choice of address has already given us something useful for free. Now ask the next question: can it also tell us which colors those stickers have?

## 00:09:13 — Use the centers to name colors

The green center sits one step along x: one, zero, zero. The white center sits one step along z: zero, zero, one. Blue is the opposite of green, so its vector is minus one, zero, zero. These are unit vectors: arrows one step long. We can use each center's fixed vector as the name of its color. Green means this direction at home, even if a green sticker later points somewhere else. The short name for the positive x unit vector is e x. Likewise e y and e z. The letter e just names a one-step arrow along an axis.

## 00:10:02 — A color has a home direction

Look at our edge at home. Draw an arrow straight out of the green sticker. It points toward the green center. Do the same for white: that arrow points up, toward the white center. An arrow perpendicular to a surface is called a normal. These normals tell us which way the stickers face. At home, they are exactly the unit vectors we just used to name green and white.

## 00:10:32 — Split the address into its ingredients

Now go back to our home vector: one, zero, one. Split its walking instructions into separate arrows. One green x arrow. No y arrow. One white z arrow. There they are: the two sticker normals. The components of the address point straight out through the cubelet's colored surfaces! This is why the coordinate choice matters. The numbers do not just locate the block. Each nonzero component also identifies a sticker and its home direction.

## 00:11:12 — A matrix can simply hold arrows

We want to keep those arrows separate, so we can follow each sticker. Write the green arrow as a column: one, zero, zero. Beside it, leave an empty column for y. Then write the white arrow: zero, zero, one. A rectangular arrangement of numbers is a matrix. Here it is simply a collection of three column vectors. Read one column at a time. Each column has x, y, and z entries, just like our address. There is no new geometry in the grid. It is a way to keep the arrows side by side.

## 00:11:57 — The diagonal comes from the address

Notice where the nonzero numbers landed: on the diagonal. One, zero, one. The same numbers as our home address! We call this the diagonal matrix of c, written diag of c. It is a simple operation: put the three coordinates on the diagonal, and fill the other entries with zeros. A nonzero column is a sticker normal. A zero column says there is no sticker on that axis. Centers, edges, and corners all fit the same rule.

## 00:12:37 — We have an address. We have its colors.

Let's collect what we know. The home address identifies a cubelet. Its nonzero coordinates count its stickers. Split that address into axis arrows, and we get the stickers' home directions. We have the static description. To answer our opening problem, we still need to move it. What happens to an arrow when the cube turns?

## 00:13:05 — Turn an arrow before writing a matrix

Take the one-step x arrow. Watch it through our top turn. It ends up pointing toward minus y. The one-step y arrow turns toward x. The upward z arrow stays up. Now think of our edge address as one x arrow plus one z arrow. Turn both arrows, and add them. We get minus y plus z. That is the new location of the edge. A rigid rotation preserves this addition. We can turn the ingredients and then add them, or add them first and turn the result. This is the property that makes linear algebra useful here.

## 00:13:50 — A matrix is a promise about three arrows

How do we record that rotation? Keep the destinations of the three one-step arrows as columns. The first column says where x went. The second says where y went. The third says where z went. This is a rotation matrix. Call this turn M. Multiplying it by a vector means: take the indicated amount of each destination arrow, and add. For one, zero, one, take the first column plus the third. The grid now has a job: it describes a motion. We have already seen that motion happen.

## 00:14:35 — Apply the motion to the home address

We will store the accumulated rotation as R. At the start, it leaves every arrow unchanged. This is called the identity matrix: one on the diagonal, zeros elsewhere. Apply R to the home vector c, and you get the current position, p. After our turn, R carries that same home address to the new location. Now use the very same motion on its stickers.

## 00:15:05 — One rotation carries every sticker

The sticker normals are the columns of our diagonal matrix. Multiplication transforms each column separately. Call the collection of current normals N. Rotate them. Green now points toward minus y. White still points up. The empty column stays empty. One multiplication gives every sticker direction at once!

## 00:15:34 — The position is inside the sticker directions

Now pause here. This is the connection that ties the whole encoding together. Add the home sticker normals, and you get c. Rotate those normals, then add them, and you get R times c. That is the cubelet's position. So the directions of its stickers already determine where it sits. One rotation keeps location and facing in agreement. That is the real payoff.

## 00:16:07 — Choose the layer geometrically

We can rotate a cubelet. Now choose which ones should move. Remember our first layer: the nine cubelets at the top. Their current z coordinate is one. The middle layer has zero; the bottom has minus one. Let v be the one-step arrow pointing out of the face we want to turn. The dot product of v with the current position reads how far that cubelet lies in this direction. For the top, it simply reads z. Positive selects the outer layer. The same test works for every face, including the negative directions. One geometric rule chooses the nine cubelets to turn.

## 00:16:55 — Follow where it is, rather than where it belongs

Here is a useful check on our story. Our edge started on the front, but the top turn carried it to the left. Its home address still identifies it. To choose the moving layer, use its current address: R times c. The two jobs are different, and the notation keeps them clear.

## 00:17:20 — A history of turns becomes one matrix

Suppose the next turn is M. Apply the old rotation first, then the new turn. That gives M times R. Update the selected cubelets. Their past moves collapse into one accumulated transformation. We can forget the sequence and keep its effect.

## 00:17:42 — Smooth motion, exact stored states

Our animation moves smoothly. Rubix stores only the completed quarter turns. A quarter turn sends each axis to another axis, perhaps reversing its sign. So the stored matrix entries stay minus one, zero, or one. No rounding is needed to represent a completed move. These matrices describe rotations without stretching or reflections. There are twenty four possible orientations.

## 00:18:17 — The home directions tell us when to stop

There is one final question. When is a cubelet solved? When every sticker points in its home direction. The transformed diagonal matrix must equal the original one. Remember the column sum? Equal sticker normals also put the cubelet at home. The same equality checks facing and location together.

## 00:18:44 — Measure exactly what the puzzle asks

And this condition has a lovely subtlety. Watch the white center spin. Its local x and y arrows turn, but the white normal still points up. Its one sticker is home, so an ordinary cube considers this center solved. Demanding the identity matrix would ask for extra information the unmarked sticker cannot show. A picture cube would need that extra orientation constraint.

## 00:19:16 — Now we can teach the computer a turn

Return to our opening problem. How do we teach the computer to turn the cube, including every sticker? Give each cubelet a home address and a rotation. The address supplies its sticker normals. The rotation carries those arrows and locates the cubelet. Read its current position to select a layer, then apply the same turn to every cubelet in it. The colors now move correctly because the geometry keeps them attached. That is the elegant choice in Rubix: a description that makes the relationships we need follow together. Finding a solution is another challenge. We have built the world in which that search happens.

## 00:20:05 — Learn to see matrices as motion

This way of seeing matrices was inspired by Grant Sanderson's Essence of Linear Algebra, from Three Blue One Brown. It is a wonderful series. Grant makes the geometry feel discoverable, and the equations feel earned. If you enjoyed this connection, I warmly recommend it. Start with a physical question. Find a description that keeps the geometry. Let the mathematics reveal what follows.

Narration is synthetic (Microsoft Edge, en-US-AndrewNeural). Original script and animation; no 3Blue1Brown footage, music, or voice.

Recommended: [Essence of Linear Algebra by Grant Sanderson / 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).
