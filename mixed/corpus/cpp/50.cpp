 */
static int update_todo(struct isl_facet_todo *first, struct isl_tab *tab)
{
	int i;
	struct isl_tab_undo *snap;
	struct isl_facet_todo *todo;

	snap = isl_tab_snap(tab);

	for (i = 0; i < tab->n_con; ++i) {
		int drop;

		if (tab->con[i].frozen)
			continue;
		if (tab->con[i].is_redundant)
			continue;

		if (isl_tab_select_facet(tab, i) < 0)
			return -1;

		todo = create_todo(tab, i);
		if (!todo)
			return -1;

		drop = has_opposite(todo, &first->next);
		if (drop < 0)
			return -1;

		if (drop)
			free_todo(todo);
		else {
			todo->next = first->next;
			first->next = todo;
		}

		if (isl_tab_rollback(tab, snap) < 0)
			return -1;
	}

	return 0;
}

#ifndef NDEBUG
static void outputScheduleInfo(llvm::raw_ostream &OS, const isl::schedule &Sched,
                               StringRef Label) {
  isl::ctx ctx = Sched.ctx();
  isl_printer *printer = isl_printer_to_str(ctx.get());
  printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
  char *str = isl_printer_print_schedule(printer, Sched.get());
  OS << Label << ": \n" << str << "\n";
  free(str);
  isl_printer_free(printer);
}

