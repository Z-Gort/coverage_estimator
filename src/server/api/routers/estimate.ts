import { z } from "zod";
import { spawn } from "child_process";
import path from "path";

import { createTRPCRouter, publicProcedure } from "~/server/api/trpc";
import { posts } from "~/server/db/schema";

export const estimateRouter = createTRPCRouter({
  create: publicProcedure
    .input(z.object({ input: z.number() }))
    .mutation(async ({ ctx, input }) => {
      // Insert row and get the ID
      const result = await ctx.db
        .insert(posts)
        .values({
          input: input.input,
        })
        .returning();

      const rowId = result[0]!.id;

      // Spawn Python script with row ID and input
      const pythonProcess = spawn(
        path.join(process.cwd(), "src/server/python/venv/bin/python"),
        [
          path.join(process.cwd(), "src/server/python/estimate.py"),
          input.input.toString(),
          rowId.toString(),
        ],
        {
          cwd: path.join(process.cwd(), "src/server/python"),
          stdio: ["pipe", "pipe", "pipe"], // Enable stdout/stderr capture
        },
      );

      // Log Python script output for debugging
      pythonProcess.stdout?.on("data", (data) => {
        console.log(`[Python stdout]: ${data}`);
      });

      pythonProcess.stderr?.on("data", (data) => {
        console.error(`[Python stderr]: ${data}`);
      });

      pythonProcess.on("close", (code) => {
        console.log(`[Python process] Exited with code: ${code}`);
      });

      pythonProcess.on("error", (error) => {
        console.error(`[Python process] Error: ${error.message}`);
      });
    }),

  getLatestResult: publicProcedure.query(async ({ ctx }) => {
    const post = await ctx.db.query.posts.findFirst({
      orderBy: (posts, { desc }) => [desc(posts.createdAt)],
    });

    return post?.result ?? null;
  }),
});
