# A cube made of transformations

## 00:00:00 — What does a computer need to remember?

Turn one face of a Rubik's cube. Now another. Fifty four colored stickers seem to demand fifty four little records. But look closer. The stickers never travel alone. They ride on small, rigid cubes. Could we remember the geometry of those cubes, and let the colors take care of themselves? Rubix does exactly that. And the surprising part is how much information one tiny home address contains.

## 00:00:31 — One cubelet. Three discoveries.

We will follow this green and white cubelet. An edge, with two stickers. First, find an address that identifies it. Second, discover how that address already knows its colors. Third, turn that description into a moving cube. By the end, we will need just two things for each cubelet: its home address, and one rotation. Let's see why.

## 00:00:58 — Start with the geometry

Put the origin at the middle of the puzzle. Replace each cubelet by a point at its center. Along each axis, there are only three levels: minus one, zero, and one. Twenty seven points. The hidden origin has no stickers, so Rubix stores the other twenty six.

## 00:01:19 — The centers give us a fixed frame

We also need directions that stay put. The six centers give us exactly that: outer face turns spin them in place without changing their positions. Choose green as positive x, red as positive y, and white as positive z. The opposite faces define the negative directions. The cube's own anatomy has handed us a coordinate system.

## 00:01:45 — Give the cubelet a permanent address

Our green and white edge belongs one step toward green and one step toward white. Its home coordinate is one, zero, one. Call it c. Now freeze that address. The cubelet can travel anywhere, but c keeps saying which cubelet it is, and where it belongs.

## 00:02:07 — Test the address: what does it give us?

Before going further, test this choice. A nonzero coordinate means a cubelet touches an outer boundary along that axis. One such coordinate gives a center. Two give an edge. Three give a corner. Add the absolute values, and you have counted its stickers. The address already knows the cubelet type!

## 00:02:30 — Can the address also know the colors?

That answers our first question: how to identify the cubelet. Now for the second. We still need its colors. Look again at one, zero, one. Could the answer already be sitting inside those three numbers?

## 00:02:47 — A color has a home direction

At home, a green sticker points toward the green center. A white sticker points toward the white center. Draw an arrow perpendicular to each sticker. These outward normals are positive x and positive z. We can identify a color by its home direction. That identity stays fixed, even when the sticker turns.

## 00:03:11 — Split the address into its ingredients

Now split the home address into its axis components. One, zero, one becomes one x arrow, no y arrow, and one z arrow. There they are. The green normal, and the white normal! The coordinate that located our cubelet also contained the directions of its stickers. Negative components work the same way: minus x identifies blue.

## 00:03:39 — Keep the arrows, instead of adding them

There is just one adjustment. Adding the arrows hides their individual identities. So keep them side by side, as columns. That gives the diagonal matrix of c. For our edge, its diagonal is one, zero, one. Green. An empty column. White. The zero is doing useful work: it says there is no sticker on that axis. A corner has three nonzero columns; a center has one.

## 00:04:11 — The same choice pays off twice

Remember how the address counted stickers? The diagonal matrix tells the same story. Its nonzero columns point along different axes. Counting those independent columns gives its rank. So rank equals the sticker count. This is why the encoding feels elegant. The physical boundaries, the cubelet type, and the color directions all agree, because they came from the same geometry.

## 00:04:39 — We have an address. We have its colors.

Two discoveries down. The home address names the cubelet, and its components identify the stickers. Our final question is motion. How can one description carry both the cubelet and its stickers through every turn?

## 00:04:56 — A matrix is a promise about three arrows

This is where linear algebra earns its place. A rotation matrix tells us where three basis arrows land. Once we know their destinations, we can transform any combination of them. Every vector follows from those three arrows. Think of the matrix as a motion you can apply, rather than a grid you must memorize.

## 00:05:20 — Follow our cubelet through one turn

Let's watch a concrete top turn. Up stays up. The x direction swings toward minus y, while y swings toward x. Our edge travels from one, zero, one to zero, minus one, one. Its home address is unchanged. The turn has changed what happens to that address.

## 00:05:44 — Apply the motion to the home address

Store the accumulated rotation as R. Apply it to the home vector c, and you get the current position, p. At the start, R is the identity. After the turn, R carries that same home address to the new location. We have recovered where the cubelet is. Now use the very same motion on its stickers.

## 00:06:07 — One rotation carries every sticker

The sticker normals are the columns of the diagonal matrix. Multiplication transforms each column separately. Rotate them. Green now points toward minus y. White still points up. The empty column stays empty. One multiplication gives every sticker direction at once!

## 00:06:30 — The position is inside the sticker directions

Now pause here. This is the connection that ties the whole encoding together. Add the home sticker normals, and you get c. Rotate those normals, then add them, and you get R times c. That is the cubelet's position. So the directions of its stickers already determine where it sits. One rotation keeps location and facing in agreement. That is the real payoff.

## 00:07:02 — Choose the layer geometrically

We can describe motion. Now we must choose which cubelets receive it. Turning a face rotates the outer layer behind it: nine cubelets together. Let v point out of that face. The dot product with p measures position along that direction. Positive means the cubelet is in that outer layer. For the top, that is simply z equals one. The geometry selects exactly the layer we wanted.

## 00:07:31 — Follow where it is, rather than where it belongs

Here is a useful check on our story. Our edge started on the front, but the top turn carried it to the left. Its home address still identifies it. To choose the moving layer, use its current address: R times c. The two jobs are different, and the notation keeps them clear.

## 00:07:53 — A history of turns becomes one matrix

Suppose the next turn is M. Apply the old rotation first, then the new turn. That gives M times R. Update the selected cubelets. Their past moves collapse into one accumulated transformation. We can forget the sequence and keep its effect.

## 00:08:13 — The idea fits directly into code

The implementation follows the same three steps. Recover the current position. Use the dot product to select the layer. Multiply the stored rotation by the new move on the left. Do this for all twenty six home address and rotation pairs, and you have the complete move rule.

## 00:08:34 — Smooth motion, exact stored states

Our animation is smooth, but a stored move is exactly a quarter turn. It only swaps coordinate axes and reverses signs. So every matrix entry stays minus one, zero, or one. These matrices remain proper rotations; there are twenty four possibilities. We get geometric motion with exact arithmetic at the endpoints.

## 00:09:00 — The home directions tell us when to stop

There is one final question. When is a cubelet solved? When every sticker points in its home direction. The transformed diagonal matrix must equal the original one. Remember the column sum? Equal sticker normals also put the cubelet at home. The same equality checks facing and location together.

## 00:09:26 — Measure exactly what the puzzle asks

And this condition has a lovely subtlety. Watch the white center spin. Its local x and y arrows turn, but the white normal still points up. Its one sticker is home, so an ordinary cube considers this center solved. Demanding the identity matrix would ask for extra information the unmarked sticker cannot show. A picture cube would need that extra orientation constraint.

## 00:09:55 — Close the loop

Four quarter turns bring every cubelet and sticker back. A turn followed by its inverse cancels too. These identities are useful tests of the model. More importantly, they let us watch the geometry close its own loop.

## 00:10:13 — What did we actually need to remember?

Return to the question we started with. What does a computer need to remember? A home address, and one rotation. The address identifies the cubelet and supplies its colors. The rotation tells us where it is and where its stickers point. Selecting a layer and applying a turn now follow from that same geometry. The search for a solution is a separate challenge; this is the small, consistent world it searches. That is Rubix's clever choice: find a description in which the facts you need emerge together.

## 00:10:51 — Learn to see matrices as motion

This way of seeing matrices was inspired by Grant Sanderson's Essence of Linear Algebra, from Three Blue One Brown. It is a wonderful series. Grant makes the geometry feel discoverable, and the equations feel earned. If you enjoyed this connection, I warmly recommend it. Look at the motion. Find the right description. Let the mathematics reveal what follows.

Narration is synthetic (Microsoft Edge, en-US-AndrewNeural). Original script and animation; no 3Blue1Brown footage, music, or voice.

Recommended: [Essence of Linear Algebra by Grant Sanderson / 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).
