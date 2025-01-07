    def add_head_section(doc, title)
      head = Nokogiri::XML::Node.new "head", doc
      title_node = Nokogiri::XML::Node.new "title", doc
      title_node.content = title
      title_node.parent = head
      css = Nokogiri::XML::Node.new "link", doc
      css["rel"] = "stylesheet"
      css["type"] = "text/css"
      css["href"] = "#{Dir.pwd}/stylesheets/epub.css"
      css.parent = head
      doc.at("body").before head
    end

        def devcontainer_options
          @devcontainer_options ||= {
            app_name: Rails.application.railtie_name.chomp("_application"),
            database: !!defined?(ActiveRecord) && database,
            active_storage: !!defined?(ActiveStorage),
            redis: !!((defined?(ActionCable) && !defined?(SolidCable)) || (defined?(ActiveJob) && !defined?(SolidQueue))),
            system_test: File.exist?("test/application_system_test_case.rb"),
            node: File.exist?(".node-version"),
            kamal: File.exist?("config/deploy.yml"),
          }
        end

        def add_constraints(reflection, key, join_ids, owner, ordered)
          scope = reflection.build_scope(reflection.aliased_table).where(key => join_ids)

          relation = reflection.klass.scope_for_association
          scope.merge!(
            relation.except(:select, :create_with, :includes, :preload, :eager_load, :joins, :left_outer_joins)
          )

          scope = reflection.constraints.inject(scope) do |memo, scope_chain_item|
            item = eval_scope(reflection, scope_chain_item, owner)
            scope.unscope!(*item.unscope_values)
            scope.where_clause += item.where_clause
            scope.order_values = item.order_values | scope.order_values
            scope
          end

