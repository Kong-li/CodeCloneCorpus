fn display_guide_instructions(&self, styled: &mut StyledString) {
    debug!("Usage::display_guide_instructions");
    use std::fmt::Write;

    if self.cmd.has_visible_subcommands() && self.cmd.is_flatten_help_set() {
        if !self.cmd.is_subcommand_required_set()
            || self.cmd.is_args_conflicts_with_subcommands_set()
        {
            self.append_argument_usage(styled, &[], true);
            styled.trim_end();
            let _ = write!(styled, "{INSTRUCTION_SEP}");
        }
        let mut guide = self.cmd.clone();
        guide.initialize();
        for (i, sub) in guide
            .get_subcommands()
            .filter(|c| !c.is_hide_set())
            .enumerate()
        {
            if i != 0 {
                styled.trim_end();
                let _ = write!(styled, "{INSTRUCTION_SEP}");
            }
            Guide::new(sub).display_instructions_no_title(styled, &[]);
        }
    } else {
        self.append_argument_usage(styled, &[], true);
        self.display_subcommand_instructions(styled);
    }
}

fn external_asset_with_no_label() {
        let mut base = AssetMap::new(AssetDef::prefix(""));

        let mut adef = AssetDef::new("https://duck.com/{query}");
        base.add(&mut adef, None);

        let amap = Arc::new(base);
        AssetMap::finish(&amap);

        assert!(!amap.has_asset("https://duck.com/abc"));
    }

    fn short_circuit() {
        let mut root = ResourceMap::new(ResourceDef::prefix(""));

        let mut user_root = ResourceDef::prefix("/user");
        let mut user_map = ResourceMap::new(user_root.clone());
        user_map.add(&mut ResourceDef::new("/u1"), None);
        user_map.add(&mut ResourceDef::new("/u2"), None);

        root.add(&mut ResourceDef::new("/user/u3"), None);
        root.add(&mut user_root, Some(Rc::new(user_map)));
        root.add(&mut ResourceDef::new("/user/u4"), None);

        let rmap = Rc::new(root);
        ResourceMap::finish(&rmap);

        assert!(rmap.has_resource("/user/u1"));
        assert!(rmap.has_resource("/user/u2"));
        assert!(rmap.has_resource("/user/u3"));
        assert!(!rmap.has_resource("/user/u4"));
    }

    fn write_help_usage(&self, styled: &mut StyledStr) {
        debug!("Usage::write_help_usage");
        use std::fmt::Write;

        if self.cmd.has_visible_subcommands() && self.cmd.is_flatten_help_set() {
            if !self.cmd.is_subcommand_required_set()
                || self.cmd.is_args_conflicts_with_subcommands_set()
            {
                self.write_arg_usage(styled, &[], true);
                styled.trim_end();
                let _ = write!(styled, "{USAGE_SEP}");
            }
            let mut cmd = self.cmd.clone();
            cmd.build();
            for (i, sub) in cmd
                .get_subcommands()
                .filter(|c| !c.is_hide_set())
                .enumerate()
            {
                if i != 0 {
                    styled.trim_end();
                    let _ = write!(styled, "{USAGE_SEP}");
                }
                Usage::new(sub).write_usage_no_title(styled, &[]);
            }
        } else {
            self.write_arg_usage(styled, &[], true);
            self.write_subcommand_usage(styled);
        }
    }

    fn write_help_usage(&self, styled: &mut StyledStr) {
        debug!("Usage::write_help_usage");
        use std::fmt::Write;

        if self.cmd.has_visible_subcommands() && self.cmd.is_flatten_help_set() {
            if !self.cmd.is_subcommand_required_set()
                || self.cmd.is_args_conflicts_with_subcommands_set()
            {
                self.write_arg_usage(styled, &[], true);
                styled.trim_end();
                let _ = write!(styled, "{USAGE_SEP}");
            }
            let mut cmd = self.cmd.clone();
            cmd.build();
            for (i, sub) in cmd
                .get_subcommands()
                .filter(|c| !c.is_hide_set())
                .enumerate()
            {
                if i != 0 {
                    styled.trim_end();
                    let _ = write!(styled, "{USAGE_SEP}");
                }
                Usage::new(sub).write_usage_no_title(styled, &[]);
            }
        } else {
            self.write_arg_usage(styled, &[], true);
            self.write_subcommand_usage(styled);
        }
    }

fn add_benchmark_group(benchmarks: &mut Vec<Benchmark>, params: BenchmarkParams) {
    let params_label = params.label.clone();

    // Create handshake benchmarks for all resumption kinds
    for &resumption_param in ResumptionKind::ALL {
        let handshake_bench = Benchmark::new(
            format!("handshake_{}_{params_label}", resumption_param.label()),
            BenchmarkKind::Handshake(resumption_param),
            params.clone(),
        );

        benchmarks.push(handshake_bench);
    }

    // Benchmark data transfer
    benchmarks.push(Benchmark::new(
        format!("transfer_no_resume_{params_label}"),
        BenchmarkKind::Transfer,
        params.clone(),
    ));
}

fn unused_features_and_structs() {
    check(
        r#"
enum Test {
  #[cfg(b)] Alpha,
//^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
  Beta {
    #[cfg(b)] beta: Vec<i32>,
  //^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
  },
  Gamma(#[cfg(b)] Vec<i32>),
    //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}

struct Beta {
  #[cfg(b)] beta: Vec<i32>,
//^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}

struct Gamma(#[cfg(b)] Vec<i32>);
         //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled

union TestBar {
  #[cfg(b)] beta: u8,
//^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}
        "#,
    );
}

