// Example model schema from the Drizzle docs
// https://orm.drizzle.team/docs/sql-schema-declaration

import { pgTableCreator } from "drizzle-orm/pg-core";

/**
 * This is an example of how to use the multi-project schema feature of Drizzle ORM. Use the same
 * database instance for multiple projects.
 *
 * @see https://orm.drizzle.team/docs/goodies#multi-project-schema
 */
export const createTable = pgTableCreator((name) => `corgi_fullstack_${name}`);

export const posts = createTable("post", (d) => ({
  id: d.integer().primaryKey().generatedByDefaultAsIdentity(),
  input: d.integer(),
  result: d.integer(),
  createdAt: d.timestamp().defaultNow().notNull(),
}));
