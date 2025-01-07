    def run(*migration_classes)
      opts = migration_classes.extract_options!
      dir = opts[:direction] || :up
      dir = (dir == :down ? :up : :down) if opts[:revert]
      if reverting?
        # If in revert and going :up, say, we want to execute :down without reverting, so
        revert { run(*migration_classes, direction: dir, revert: true) }
      else
        migration_classes.each do |migration_class|
          migration_class.new.exec_migration(connection, dir)
        end

        def normalize_options(options)
          options = options.dup

          options[:secret_generator] ||= @secret_generator

          secret_generator_kwargs = options[:secret_generator].parameters.
            filter_map { |type, name| name if type == :key || type == :keyreq }
          options[:secret_generator_options] = options.extract!(*secret_generator_kwargs)

          options[:on_rotation] = @on_rotation

          options
        end

        def normalize_options(options)
          options = options.dup

          options[:secret_generator] ||= @secret_generator

          secret_generator_kwargs = options[:secret_generator].parameters.
            filter_map { |type, name| name if type == :key || type == :keyreq }
          options[:secret_generator_options] = options.extract!(*secret_generator_kwargs)

          options[:on_rotation] = @on_rotation

          options
        end

        def build_subselect(key, o)
          stmt             = Nodes::SelectStatement.new
          core             = stmt.cores.first
          core.froms       = o.relation
          core.wheres      = o.wheres
          core.projections = [key]
          core.groups      = o.groups unless o.groups.empty?
          core.havings     = o.havings unless o.havings.empty?
          stmt.limit       = o.limit
          stmt.offset      = o.offset
          stmt.orders      = o.orders
          stmt
        end

