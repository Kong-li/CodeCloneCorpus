        def _update_row(attribute_names, attempted_action = "update")
          return super unless locking_enabled?

          begin
            locking_column = self.class.locking_column
            lock_attribute_was = @attributes[locking_column]

            update_constraints = _query_constraints_hash

            attribute_names = attribute_names.dup if attribute_names.frozen?
            attribute_names << locking_column

            if self[locking_column].nil?
              raise(<<-MSG.squish)
                For optimistic locking, '#{locking_column}' should not be set to `nil`/`NULL`.
                Are you missing a default value or validation on '#{locking_column}'?
              MSG
            end

      def source_extract(indentation = 0)
        return [] unless num = line_number
        num = num.to_i

        source_code = @template.encode!.split("\n")

        start_on_line = [ num - SOURCE_CODE_RADIUS - 1, 0 ].max
        end_on_line   = [ num + SOURCE_CODE_RADIUS - 1, source_code.length].min

        indent = end_on_line.to_s.size + indentation
        return [] unless source_code = source_code[start_on_line..end_on_line]

        formatted_code_for(source_code, start_on_line, indent)
      end

  def write_app_file(options={})
    options[:routes] ||= ['get("/foo") { erb :foo }']
    options[:inline_templates] ||= nil
    options[:extensions] ||= []
    options[:middlewares] ||= []
    options[:filters] ||= []
    options[:errors] ||= {}
    options[:name] ||= app_name
    options[:enable_reloader] = true unless options[:enable_reloader] === false
    options[:parent] ||= 'Sinatra::Base'

    update_file(app_file_path) do |f|
      template_path = File.expand_path('reloader/app.rb.erb', __dir__)
      template = Tilt.new(template_path, nil, :trim => '<>')
      f.write template.render(Object.new, options)
    end

