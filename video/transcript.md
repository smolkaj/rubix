# A cube made of transformations

## 00:00:00 — A cube made of transformations

A Rubik's cube looks like a puzzle about colors. But watch what happens when a face turns. Each little piece travels through space, and its stickers turn with it. What if our program remembered that geometry directly? This is the idea behind Rubix.

## 00:00:17 — Where is it? Which way does it face?

Follow just this green and white edge. To describe it, we need to answer two questions. Where is the piece? And which way do its two stickers point? Rubix answers both questions with the same rotation matrix. To see why that works, let's first give the cube a coordinate system.

## 00:00:36 — Let the centers define the axes

Place the origin at the middle of the cube. In Rubix, positive x points toward the green center, positive y toward red, and positive z toward white. Their opposite directions are blue, orange, and yellow. We keep this frame fixed. Face turns spin the centers in place, but do not move them to other faces.

## 00:00:58 — Replace pieces with their centers

Now replace every little piece by a point at its center. Along each axis, there are only three possible coordinates: minus one, zero, and one. This gives a three by three by three lattice. The point at the origin has no stickers, so Rubix leaves it out. Twenty six points remain.

## 00:01:19 — Coordinates already know the piece type

A nonzero coordinate means the piece reaches an outer face along that axis. Count the nonzero coordinates, and you count its stickers. One gives a center. Two gives an edge. Three gives a corner. The geometry already contains the information we might otherwise encode as a separate piece type.

## 00:01:39 — A permanent name for a moving piece

For our green and white edge, the solved coordinate is one, zero, one. Call this vector c. Here comes an important distinction: c is a permanent name for this particular piece. Even after a scramble, c never changes. It says where the piece belongs, rather than where it happens to be now.

## 00:02:00 — A matrix tells us where the axes land

Beside c, store a rotation matrix R. Think of its columns as three arrows: where the original x, y, and z directions have landed. A matrix is a description of a transformation. Once we know what it does to these three basis vectors, we know what it does to every vector.

## 00:02:21 — Carry the home vector through the turn

In particular, we know what it does to c. Multiplying R by c gives the piece's current position, p. At the start, R is the identity, so p equals c. After a turn, R carries the home position to a new point. We do not need to store a second, independently updated position.

## 00:02:42 — One concrete quarter turn

Take this top face move from Rubix. It leaves z alone, sends x to minus y, and sends y to x. Our edge travels from one, zero, one to zero, minus one, one. The continuous arc is only an animation. The program stores the exact integer result of the quarter turn.

## 00:03:04 — Read the matrix as three arrow destinations

Look at the columns of the move matrix. The first is zero, minus one, zero: the destination of the x arrow. The second is one, zero, zero: the destination of the y arrow. The third keeps the z arrow fixed. Matrix multiplication combines these destinations using the coordinates of our vector.

## 00:03:28 — A sticker is a direction

Now for the stickers. Represent a sticker by its outward normal: an arrow perpendicular to its surface. Green's permanent color identity is positive x, its direction in the solved cube. After our top turn, that same green sticker points toward negative y. Its identity stays green; its current direction changes.

## 00:03:50 — Split the home vector into sticker normals

Why does the home coordinate tell us which stickers the piece owns? Split c into its three axis components. Our edge gives one x arrow, no y arrow, and one z arrow. These are exactly its green and white sticker normals at home. Negative components work too: minus x identifies a blue sticker.

## 00:04:12 — Keep those three components as columns

Rather than adding the component arrows, put them side by side as columns. That is the diagonal matrix of c, which we'll call D. For this edge, its diagonal is one, zero, one. The zero column represents an absent sticker. For a corner all three columns are present. For a center only one is.

## 00:04:34 — Rotate every sticker in one multiplication

Multiplication acts on each column separately. So R times D rotates all of the sticker normals at once. The green column now points along negative y. The white column still points up. The zero column stays zero. The very same R that moved the piece also describes where every sticker faces.

## 00:04:55 — Position is already inside the normals

There is an especially satisfying connection here. Add the columns of D and you get c. Rotate them and add again, and you get R c, the current position. So the transformed sticker normals also determine where the piece sits. Location and orientation are two views of one geometric description.

## 00:05:17 — Which pieces should turn?

A face turn should affect only one layer. Let v point outward from that face. The dot product v dot p measures the piece's coordinate along that direction. Select the pieces for which it is positive. For the top face, that means z equals one. Exactly nine pieces pass the test.

## 00:05:37 — Select by current position

The test must use the current position, R c. After our top turn, the green and white edge lives on the left face, even though its home coordinate was on the front. Selecting by c would move the wrong pieces. Selecting by R c automatically follows every piece through the scramble.

## 00:05:56 — Apply the next turn after the old one

For each selected piece, update R to M times R. The old R acts first, taking the piece from home to its current state. The new move M acts next. That order explains why M goes on the left. Matrix multiplication becomes the record of a piece's entire history of turns.

## 00:06:17 — The geometric rule is the code

And here is the whole update in the language of the implementation. Compute the current position. Use a dot product to test whether the move applies. If it does, compose the move matrix with the stored rotation. Otherwise keep that rotation. Repeat this rule over the twenty six pairs that make up the cube.

## 00:06:37 — The animation is smooth. The state is exact.

The animation passes through arbitrary angles, but stored states never do. Quarter turns only permute the coordinate axes and reverse some signs. Their matrices have entries minus one, zero, and one. They stay orthogonal, with determinant plus one. There are just twenty four such proper rotations of a cube. Exact endpoints avoid accumulated rounding error.

## 00:07:03 — Four quarter turns close the loop

This gives us useful checks as well as an elegant model. Four quarter turns return every piece and sticker to the start. A move followed by its inverse cancels. These identities test the geometry as a whole, rather than checking a few hand written sticker permutations.

## 00:07:22 — Every sticker points home

When is a piece solved? When every sticker normal points in its original direction. That is precisely R times the diagonal of c equals the diagonal of c. Because position is the sum of those columns, this equality also puts the piece at home. Rubix checks this condition for every piece.

## 00:07:43 — Solved does not always mean R = I

Why not simply demand that R be the identity? Watch the white center during a top turn. Its little local x and y arrows rotate, but its only sticker still points up. For an ordinary cube, that center is solved. Rubix correctly ignores the orientation of an unmarked center within its own face. A picture cube would need a stricter condition.

## 00:08:07 — One geometric story, four operations

We now have a complete description of the puzzle's state and motion. A home vector identifies the piece. A rotation gives its position and sticker directions. A dot product chooses a layer. Multiplication applies the move. And an equality recognizes solved pieces. The search for a solution is a separate problem; this encoding supplies its consistent geometric world.

## 00:08:33 — Learn to see matrices as motion

If this way of seeing matrices appeals to you, watch Grant Sanderson's Essence of Linear Algebra, from Three Blue One Brown. It is a superb series: patient, visual, and beautifully clear about why the mathematics works. It directly inspired Rubix. This original explanation follows that geometric spirit. Once you see a matrix as motion, a cube of colored stickers becomes a small world of vectors and transformations.

Narration is synthetic (Microsoft Edge, en-US-AndrewNeural). Original script and animation; no 3Blue1Brown footage, music, or voice.

Recommended: [Essence of Linear Algebra by Grant Sanderson / 3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab).
