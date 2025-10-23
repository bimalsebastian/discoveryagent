# Databricks notebook source
# MAGIC %sql
# MAGIC create or replace view yourcatalog.data_discovery.table_schema_tags
# MAGIC as
# MAGIC select * from yourcatalog.information_schema.table_tags
# MAGIC union all
# MAGIC select catalog_name,schema_name, null as table_name, tag_name, tag_value from your_catalog.information_schema.schema_tags;
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,IDENTITY & OWNERSHIP
# MAGIC %sql
# MAGIC -- DOMAIN 1: IDENTITY & OWNERSHIP
# MAGIC -- Focus: WHO owns, manages, or is responsible for data products
# MAGIC
# MAGIC CREATE OR REPLACE TABLE your_catalog.yourschema.metadata_identity_source
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     array_join(collect_list(DISTINCT catalog_name), ', ') AS catalog_name,
# MAGIC     schema_name,
# MAGIC     collect_list(DISTINCT table_name) AS tables_in_product,
# MAGIC     -- Identity tags from our enrichment script
# MAGIC     max(CASE WHEN tag_name = 'identity_primary_owner' THEN tag_value END) AS primary_owner,
# MAGIC     max(CASE WHEN tag_name = 'identity_backup_owner' THEN tag_value END) AS backup_owner,
# MAGIC     max(CASE WHEN tag_name = 'identity_steward_squad' THEN tag_value END) AS steward_squad,
# MAGIC     max(CASE WHEN tag_name = 'identity_responsible_team' THEN tag_value END) AS responsible_team,
# MAGIC     max(CASE WHEN tag_name = 'identity_contact_email' THEN tag_value END) AS contact_email,
# MAGIC     max(CASE WHEN tag_name = 'identity_data_steward' THEN tag_value END) AS data_steward,
# MAGIC     max(CASE WHEN tag_name = 'identity_managing_squad' THEN tag_value END) AS managing_squad,
# MAGIC     max(CASE WHEN tag_name = 'business_data_product_title' THEN tag_value END) AS data_product_title,
# MAGIC     max(CASE WHEN tag_name = 'business_project_name' THEN tag_value END) AS project_name,
# MAGIC     count(DISTINCT table_name) AS table_count
# MAGIC   FROM your_catalog.yourschema.table_schema_tags
# MAGIC   GROUP BY schema_name
# MAGIC )
# MAGIC SELECT concat(
# MAGIC   "IDENTITY & OWNERSHIP\n",
# MAGIC   "===================\n\n",
# MAGIC   
# MAGIC   "QUICK FACTS:\n",
# MAGIC   "Schema: ", schema_name, "\n",
# MAGIC   "Owner: ", coalesce(primary_owner, 'Not Assigned'), "\n",
# MAGIC   "Squad: ", coalesce(steward_squad, managing_squad, 'Not Assigned'), "\n",
# MAGIC   "Contact: ", coalesce(contact_email, 'Not Available'), "\n\n",
# MAGIC   
# MAGIC   "DATA PRODUCT: ", coalesce(data_product_title, schema_name), "\n",
# MAGIC   "LOCATION: ", catalog_name, ".", schema_name, "\n",
# MAGIC   "CONTAINS: ", table_count, " tables\n",
# MAGIC   "TABLES: ", array_join(tables_in_product, ', '), "\n\n",
# MAGIC   
# MAGIC   "PRIMARY OWNERSHIP:\n",
# MAGIC   "- PRIMARY DATA OWNER: ", coalesce(primary_owner, 'Not Assigned'), "\n",
# MAGIC   "- BACKUP OWNER: ", coalesce(backup_owner, 'Not Assigned'), "\n",
# MAGIC   "- DATA STEWARD: ", coalesce(data_steward, 'Not Assigned'), "\n\n",
# MAGIC   
# MAGIC   "TEAM RESPONSIBILITY:\n",
# MAGIC   "- MANAGING SQUAD: ", coalesce(managing_squad, steward_squad, 'Not Assigned'), "\n",
# MAGIC   "- STEWARD SQUAD: ", coalesce(steward_squad, 'Not Assigned'), "\n",
# MAGIC   "- RESPONSIBLE TEAM: ", coalesce(responsible_team, 'Not Assigned'), "\n\n",
# MAGIC   
# MAGIC   "CONTACT INFORMATION:\n",
# MAGIC   "- EMAIL CONTACT: ", coalesce(contact_email, 'Not Available'), "\n",
# MAGIC   "- For data issues: ", coalesce(primary_owner, 'Unknown'), "\n",
# MAGIC   "- For support: ", coalesce(responsible_team, steward_squad, 'Unknown'), "\n\n",
# MAGIC   
# MAGIC   "OWNERSHIP QUERIES (Natural Language):\n",
# MAGIC   "Q: Who owns ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(primary_owner, 'Not assigned'), "\n\n",
# MAGIC   
# MAGIC   "Q: Who is responsible for ", schema_name, "?\n", 
# MAGIC   "A: ", coalesce(responsible_team, steward_squad, primary_owner, 'Not assigned'), "\n\n",
# MAGIC   
# MAGIC   "Q: Who manages ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(managing_squad, steward_squad, 'Not assigned'), "\n\n",
# MAGIC   
# MAGIC   "Q: Contact for ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(contact_email, primary_owner, 'Not available'), "\n\n",
# MAGIC   
# MAGIC   "OWNERSHIP RELATIONSHIPS:\n",
# MAGIC   coalesce(primary_owner, 'unknown'), " OWNS ", schema_name, "\n",
# MAGIC   coalesce(steward_squad, 'unknown'), " MANAGES ", schema_name, "\n",
# MAGIC   coalesce(responsible_team, 'unknown'), " MAINTAINS ", schema_name, "\n",
# MAGIC   coalesce(data_steward, 'unknown'), " STEWARDS ", schema_name, "\n",
# MAGIC   schema_name, " BELONGS_TO_PROJECT ", coalesce(project_name, 'unknown'), "\n",
# MAGIC   schema_name, " CONTACT_EMAIL ", coalesce(contact_email, 'none'), "\n\n",
# MAGIC   
# MAGIC   "KEYWORDS: ",
# MAGIC   "owner ", coalesce(primary_owner, ''), " ",
# MAGIC   "owns ", coalesce(primary_owner, ''), " ",
# MAGIC   "contact ", coalesce(contact_email, ''), " ",
# MAGIC   "responsible ", coalesce(responsible_team, ''), " ",
# MAGIC   "manages ", coalesce(managing_squad, ''), " ",
# MAGIC   "squad ", coalesce(steward_squad, ''), " ",
# MAGIC   "team ", coalesce(responsible_team, ''), " ",
# MAGIC   "steward ", coalesce(data_steward, ''), " ",
# MAGIC   "accountability ", coalesce(steward_squad, ''), " "
# MAGIC ) AS document_text,
# MAGIC schema_name AS primary_key
# MAGIC FROM base;

# COMMAND ----------

# DBTITLE 1,BUSINESS CONTEXT & STRUCTURE DOMAIN
# MAGIC %sql
# MAGIC -- DOMAIN 2: BUSINESS CONTEXT & STRUCTURE  
# MAGIC -- Focus: WHAT projects, domains, business context, organizational structure
# MAGIC CREATE OR REPLACE TABLE your_catalog.yourschema.metadata_business_source
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     array_join(collect_list(DISTINCT catalog_name), ', ') AS catalog_name,
# MAGIC     schema_name,
# MAGIC     collect_list(DISTINCT table_name) AS tables_in_product,
# MAGIC     -- Business context tags from our enrichment script
# MAGIC     max(CASE WHEN tag_name = 'business_business_domain' THEN tag_value END) AS business_domain,
# MAGIC     max(CASE WHEN tag_name = 'business_project_name' THEN tag_value END) AS project_name,
# MAGIC     max(CASE WHEN tag_name = 'business_business_purpose' THEN tag_value END) AS business_purpose,
# MAGIC     max(CASE WHEN tag_name = 'business_use_case' THEN tag_value END) AS use_case,
# MAGIC     max(CASE WHEN tag_name = 'business_geographic_scope' THEN tag_value END) AS geographic_scope,
# MAGIC     max(CASE WHEN tag_name = 'business_business_entity' THEN tag_value END) AS business_entity,
# MAGIC     max(CASE WHEN tag_name = 'business_organizational_unit' THEN tag_value END) AS organizational_unit,
# MAGIC     max(CASE WHEN tag_name = 'business_business_criticality' THEN tag_value END) AS business_criticality,
# MAGIC     max(CASE WHEN tag_name = 'business_data_product_title' THEN tag_value END) AS data_product_title,
# MAGIC     max(CASE WHEN tag_name = 'business_project_context' THEN tag_value END) AS project_context,
# MAGIC     max(CASE WHEN tag_name = 'business_business_use_case' THEN tag_value END) AS business_use_case,
# MAGIC     max(CASE WHEN tag_name = 'business_domain_area' THEN tag_value END) AS domain_area,
# MAGIC     max(CASE WHEN tag_name = 'business_business_value' THEN tag_value END) AS business_value,
# MAGIC     count(DISTINCT table_name) AS table_count
# MAGIC   FROM your_catalog.yourschema.table_schema_tags
# MAGIC   GROUP BY schema_name
# MAGIC )
# MAGIC SELECT concat(
# MAGIC   "BUSINESS CONTEXT & STRUCTURE\n",
# MAGIC   "============================\n\n",
# MAGIC   
# MAGIC   "QUICK FACTS:\n",
# MAGIC   "Schema: ", schema_name, "\n",
# MAGIC   "Business Domain: ", coalesce(business_domain, 'Not Specified'), "\n",
# MAGIC   "Project: ", coalesce(project_name, 'Not Specified'), "\n",
# MAGIC   "Criticality: ", coalesce(business_criticality, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "DATA PRODUCT: ", coalesce(data_product_title, schema_name), "\n",
# MAGIC   "LOCATION: ", catalog_name, ".", schema_name, "\n",
# MAGIC   "SIZE: ", table_count, " tables\n",
# MAGIC   "TABLES: ", array_join(tables_in_product, ', '), "\n\n",
# MAGIC   
# MAGIC   "PROJECT INFORMATION:\n",
# MAGIC   "- PROJECT NAME: ", coalesce(project_name, 'Not Specified'), "\n",
# MAGIC   "- PROJECT CONTEXT: ", coalesce(project_context, 'Not Specified'), "\n",
# MAGIC   "- BUSINESS PURPOSE: ", coalesce(business_purpose, 'Not Specified'), "\n",
# MAGIC   "- BUSINESS VALUE: ", coalesce(business_value, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "USE CASE DETAILS:\n",
# MAGIC   "- PRIMARY USE CASE: ", coalesce(business_use_case, use_case, 'Not Specified'), "\n",
# MAGIC   "- DOMAIN AREA: ", coalesce(domain_area, 'Not Specified'), "\n",
# MAGIC   "- APPLICATION: ", schema_name, " is used for ", coalesce(business_use_case, use_case, 'general business purposes'), "\n",
# MAGIC   "- SUPPORTS: ", coalesce(use_case, 'business operations'), "\n\n",
# MAGIC   
# MAGIC   "BUSINESS ORGANIZATION:\n",
# MAGIC   "- BUSINESS DOMAIN: ", coalesce(business_domain, 'Not Specified'), "\n",
# MAGIC   "- BUSINESS ENTITY: ", coalesce(business_entity, 'Not Specified'), "\n",
# MAGIC   "- ORGANIZATIONAL UNIT: ", coalesce(organizational_unit, 'Not Specified'), "\n",
# MAGIC   "- BUSINESS CRITICALITY: ", coalesce(business_criticality, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "GEOGRAPHIC SCOPE:\n",
# MAGIC   "- COVERAGE: ", coalesce(geographic_scope, 'Not Specified'), "\n",
# MAGIC   "- REGION: ", coalesce(geographic_scope, 'Global'), "\n\n",
# MAGIC   
# MAGIC   "BUSINESS QUERIES (Natural Language):\n",
# MAGIC   "Q: What project is ", schema_name, " for?\n",
# MAGIC   "A: ", coalesce(project_name, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "Q: What is the business purpose of ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(business_purpose, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "Q: What business domain is ", schema_name, " in?\n",
# MAGIC   "A: ", coalesce(business_domain, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "Q: What is ", schema_name, " used for?\n",
# MAGIC   "A: ", coalesce(business_use_case, use_case, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "BUSINESS RELATIONSHIPS:\n",
# MAGIC   schema_name, " BELONGS_TO_PROJECT ", coalesce(project_name, 'unknown'), "\n",
# MAGIC   schema_name, " IN_BUSINESS_DOMAIN ", coalesce(business_domain, 'unknown'), "\n",
# MAGIC   schema_name, " SERVES_ENTITY ", coalesce(business_entity, 'unknown'), "\n",
# MAGIC   schema_name, " COVERS_GEOGRAPHY ", coalesce(geographic_scope, 'unknown'), "\n",
# MAGIC   schema_name, " SUPPORTS_USECASE ", coalesce(business_use_case, 'unknown'), "\n",
# MAGIC   schema_name, " HAS_CRITICALITY ", coalesce(business_criticality, 'unknown'), "\n",
# MAGIC   schema_name, " PROVIDES_VALUE ", coalesce(business_value, 'unknown'), "\n\n",
# MAGIC   
# MAGIC   "KEYWORDS: ",
# MAGIC   "project ", coalesce(project_name, ''), " ",
# MAGIC   "domain ", coalesce(business_domain, ''), " ",
# MAGIC   "business ", coalesce(business_entity, ''), " ",
# MAGIC   "usecase ", coalesce(business_use_case, ''), " ",
# MAGIC   "purpose ", coalesce(business_purpose, ''), " ",
# MAGIC   "entity ", coalesce(business_entity, ''), " ",
# MAGIC   "geographic ", coalesce(geographic_scope, ''), " ",
# MAGIC   "criticality ", coalesce(business_criticality, ''), " ",
# MAGIC   "value ", coalesce(business_value, ''), " "
# MAGIC ) AS document_text,
# MAGIC schema_name AS primary_key  
# MAGIC FROM base;

# COMMAND ----------

# DBTITLE 1,TECHNICAL ARCHITECTURE DOMAIN
# MAGIC %sql
# MAGIC -- DOMAIN 3: TECHNICAL ARCHITECTURE
# MAGIC -- Focus: HOW data is structured, technical implementation, system details
# MAGIC CREATE OR REPLACE TABLE bimal_data_discovery.data_discovery.metadata_technical_source
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     array_join(collect_list(DISTINCT catalog_name), ', ') AS catalog_name,
# MAGIC     schema_name,
# MAGIC     collect_list(DISTINCT table_name) AS tables_in_product,
# MAGIC     -- Technical tags from our enrichment script
# MAGIC     max(CASE WHEN tag_name = 'technical_technical_architecture' THEN tag_value END) AS technical_architecture,
# MAGIC     max(CASE WHEN tag_name = 'technical_data_source_system' THEN tag_value END) AS data_source_system,
# MAGIC     max(CASE WHEN tag_name = 'technical_update_frequency' THEN tag_value END) AS update_frequency,
# MAGIC     max(CASE WHEN tag_name = 'technical_data_volume_gb' THEN tag_value END) AS data_volume_gb,
# MAGIC     max(CASE WHEN tag_name = 'technical_table_count' THEN tag_value END) AS expected_table_count,
# MAGIC     max(CASE WHEN tag_name = 'technical_integration_method' THEN tag_value END) AS integration_method,
# MAGIC     max(CASE WHEN tag_name = 'technical_catalog_tier' THEN tag_value END) AS catalog_tier,
# MAGIC     max(CASE WHEN tag_name = 'technical_table_type' THEN tag_value END) AS table_type,
# MAGIC     max(CASE WHEN tag_name = 'technical_source_system' THEN tag_value END) AS source_system,
# MAGIC     max(CASE WHEN tag_name = 'technical_refresh_pattern' THEN tag_value END) AS refresh_pattern,
# MAGIC     max(CASE WHEN tag_name = 'technical_partitioning' THEN tag_value END) AS partitioning,
# MAGIC     max(CASE WHEN tag_name = 'technical_clustering' THEN tag_value END) AS clustering,
# MAGIC     max(CASE WHEN tag_name = 'business_data_product_title' THEN tag_value END) AS data_product_title,
# MAGIC     count(DISTINCT table_name) AS actual_table_count
# MAGIC   FROM bimal_data_discovery.data_discovery.table_schema_tags
# MAGIC   GROUP BY schema_name
# MAGIC )
# MAGIC SELECT concat(
# MAGIC   "TECHNICAL ARCHITECTURE\n",
# MAGIC   "=====================\n\n",
# MAGIC   
# MAGIC   "QUICK FACTS:\n",
# MAGIC   "Schema: ", schema_name, "\n",
# MAGIC   "Source System: ", coalesce(source_system, data_source_system, 'Not Specified'), "\n",
# MAGIC   "Update Frequency: ", coalesce(update_frequency, refresh_pattern, 'Not Specified'), "\n",
# MAGIC   "Table Count: ", actual_table_count, "\n\n",
# MAGIC   
# MAGIC   "DATA PRODUCT: ", coalesce(data_product_title, schema_name), "\n",
# MAGIC   "FULL LOCATION: ", catalog_name, ".", schema_name, "\n",
# MAGIC   "CATALOG TIER: ", coalesce(catalog_tier, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "TECHNICAL STRUCTURE:\n",
# MAGIC   "- ARCHITECTURE PATTERN: ", coalesce(technical_architecture, 'Not Specified'), "\n",
# MAGIC   "- TABLE COUNT: ", actual_table_count, " tables\n",
# MAGIC   "- DATA VOLUME: ", coalesce(data_volume_gb, 'Not Specified'), " GB\n",
# MAGIC   "- TABLE TYPE: ", coalesce(table_type, 'Mixed'), "\n\n",
# MAGIC   
# MAGIC   "TABLE INVENTORY:\n",
# MAGIC   "Tables in ", schema_name, ": ", array_join(tables_in_product, ', '), "\n\n",
# MAGIC   
# MAGIC   "TABLE DETAILS:\n",
# MAGIC   concat_ws('\n', transform(tables_in_product, t -> concat(
# MAGIC     "  • ", catalog_name, ".", schema_name, ".", t,
# MAGIC     " (source: ", coalesce(source_system, data_source_system, 'unknown'), ")"
# MAGIC   ))), "\n\n",
# MAGIC   
# MAGIC   "SYSTEM INTEGRATION:\n",
# MAGIC   "- PRIMARY SOURCE SYSTEM: ", coalesce(source_system, data_source_system, 'Not Specified'), "\n",
# MAGIC   "- INTEGRATION METHOD: ", coalesce(integration_method, 'Not Specified'), "\n",
# MAGIC   "- UPDATE FREQUENCY: ", coalesce(update_frequency, refresh_pattern, 'Not Specified'), "\n",
# MAGIC   "- REFRESH PATTERN: ", coalesce(refresh_pattern, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "DATA OPTIMIZATION:\n",
# MAGIC   "- PARTITIONING: ", coalesce(partitioning, 'Not Specified'), "\n",
# MAGIC   "- CLUSTERING: ", coalesce(clustering, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "TECHNICAL QUERIES (Natural Language):\n",
# MAGIC   "Q: What source system feeds ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(source_system, data_source_system, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "Q: What tables are in ", schema_name, "?\n",
# MAGIC   "A: ", array_join(tables_in_product, ', '), "\n\n",
# MAGIC   
# MAGIC   "Q: How often is ", schema_name, " refreshed?\n",
# MAGIC   "A: ", coalesce(update_frequency, refresh_pattern, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "Q: What is the technical architecture of ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(technical_architecture, 'Not specified'), "\n\n",
# MAGIC   
# MAGIC   "TECHNICAL RELATIONSHIPS:\n",
# MAGIC   schema_name, " CONTAINS_TABLES ", actual_table_count, "_tables\n",
# MAGIC   schema_name, " SOURCED_FROM ", coalesce(source_system, data_source_system, 'unknown'), "\n",
# MAGIC   schema_name, " REFRESHED_VIA ", coalesce(integration_method, 'unknown'), "\n",
# MAGIC   schema_name, " UPDATED_FREQUENCY ", coalesce(update_frequency, 'unknown'), "\n",
# MAGIC   schema_name, " ARCHITECTURE_PATTERN ", coalesce(technical_architecture, 'unknown'), "\n",
# MAGIC   schema_name, " CATALOG_TIER ", coalesce(catalog_tier, 'unknown'), "\n\n",
# MAGIC   
# MAGIC   "KEYWORDS: ",
# MAGIC   "catalog ", catalog_name, " ",
# MAGIC   "schema ", schema_name, " ",
# MAGIC   "tables ", array_join(tables_in_product, ' '), " ",
# MAGIC   "source ", coalesce(source_system, data_source_system, ''), " ",
# MAGIC   "system ", coalesce(data_source_system, ''), " ",
# MAGIC   "refresh ", coalesce(refresh_pattern, ''), " ",
# MAGIC   "frequency ", coalesce(update_frequency, ''), " ",
# MAGIC   "technical ", coalesce(technical_architecture, ''), " ",
# MAGIC   "architecture ", coalesce(technical_architecture, ''), " ",
# MAGIC   "integration ", coalesce(integration_method, ''), " ",
# MAGIC   "contains ", schema_name, " "
# MAGIC ) AS document_text,
# MAGIC schema_name AS primary_key
# MAGIC FROM base;

# COMMAND ----------

# DBTITLE 1,GOVERNANCE RELATIONSHIPS
# MAGIC %sql
# MAGIC -- DOMAIN 4: GOVERNANCE & COMPLIANCE
# MAGIC -- Focus: Compliance, classifications, policies, data governance
# MAGIC
# MAGIC CREATE OR REPLACE TABLE your_catalog.yourschema.metadata_governance_source
# MAGIC TBLPROPERTIES (delta.enableChangeDataFeed = true)
# MAGIC AS
# MAGIC WITH base AS (
# MAGIC   SELECT 
# MAGIC     array_join(collect_list(DISTINCT catalog_name), ', ') AS catalog_name,
# MAGIC     schema_name,
# MAGIC     collect_list(DISTINCT table_name) AS tables_in_product,
# MAGIC     -- Governance tags from our enrichment script
# MAGIC     max(CASE WHEN tag_name = 'governance_data_classification' THEN tag_value END) AS data_classification,
# MAGIC     max(CASE WHEN tag_name = 'governance_sensitivity_level' THEN tag_value END) AS sensitivity_level,
# MAGIC     max(CASE WHEN tag_name = 'governance_sensitivity' THEN tag_value END) AS sensitivity,
# MAGIC     max(CASE WHEN tag_name = 'governance_data_criticality' THEN tag_value END) AS data_criticality,
# MAGIC     max(CASE WHEN tag_name = 'governance_criticality' THEN tag_value END) AS criticality,
# MAGIC     max(CASE WHEN tag_name = 'governance_retention_policy' THEN tag_value END) AS retention_policy,
# MAGIC     max(CASE WHEN tag_name = 'governance_retention_years' THEN tag_value END) AS retention_years,
# MAGIC     max(CASE WHEN tag_name = 'governance_compliance_frameworks' THEN tag_value END) AS compliance_frameworks,
# MAGIC     max(CASE WHEN tag_name = 'governance_regulatory_scope' THEN tag_value END) AS regulatory_scope,
# MAGIC     max(CASE WHEN tag_name = 'governance_access_control_level' THEN tag_value END) AS access_control_level,
# MAGIC     max(CASE WHEN tag_name = 'governance_pii_content' THEN tag_value END) AS pii_content,
# MAGIC     max(CASE WHEN tag_name = 'governance_contains_pii' THEN tag_value END) AS contains_pii,
# MAGIC     max(CASE WHEN tag_name = 'governance_encryption_required' THEN tag_value END) AS encryption_required,
# MAGIC     max(CASE WHEN tag_name = 'governance_data_quality_tier' THEN tag_value END) AS data_quality_tier,
# MAGIC     max(CASE WHEN tag_name = 'business_data_product_title' THEN tag_value END) AS data_product_title,
# MAGIC     max(CASE WHEN tag_name = 'identity_data_steward' THEN tag_value END) AS data_steward,
# MAGIC     max(CASE WHEN tag_name = 'identity_steward_squad' THEN tag_value END) AS steward_squad,
# MAGIC     count(DISTINCT table_name) AS table_count
# MAGIC   FROM your_catalog.yourschema.table_schema_tags
# MAGIC   GROUP BY schema_name
# MAGIC )
# MAGIC SELECT concat(
# MAGIC   "GOVERNANCE & COMPLIANCE\n",
# MAGIC   "======================\n\n",
# MAGIC   
# MAGIC   "QUICK FACTS:\n",
# MAGIC   "Schema: ", schema_name, "\n",
# MAGIC   "Classification: ", coalesce(data_classification, 'Not Classified'), "\n",
# MAGIC   "Sensitivity: ", coalesce(sensitivity, sensitivity_level, 'Not Specified'), "\n",
# MAGIC   "Criticality: ", coalesce(criticality, data_criticality, 'Not Assessed'), "\n\n",
# MAGIC   
# MAGIC   "DATA PRODUCT: ", coalesce(data_product_title, schema_name), "\n",
# MAGIC   "LOCATION: ", catalog_name, ".", schema_name, "\n",
# MAGIC   "GOVERNANCE SCOPE: ", table_count, " tables\n",
# MAGIC   "TABLES: ", array_join(tables_in_product, ', '), "\n\n",
# MAGIC   
# MAGIC   "DATA CLASSIFICATION & SECURITY:\n",
# MAGIC   "- CLASSIFICATION: ", coalesce(data_classification, 'Not Classified'), "\n",
# MAGIC   "- SENSITIVITY LEVEL: ", coalesce(sensitivity, sensitivity_level, 'Not Specified'), "\n",
# MAGIC   "- CRITICALITY RATING: ", coalesce(criticality, data_criticality, 'Not Assessed'), "\n",
# MAGIC   "- ACCESS CONTROL: ", coalesce(access_control_level, 'Not Defined'), "\n",
# MAGIC   "- ENCRYPTION REQUIRED: ", coalesce(encryption_required, 'Not Specified'), "\n",
# MAGIC   "- PII CONTENT: ", coalesce(contains_pii, pii_content, 'None'), "\n\n",
# MAGIC   
# MAGIC   "LIFECYCLE & RETENTION:\n",
# MAGIC   "- RETENTION POLICY: ", coalesce(retention_policy, 'Not Defined'), "\n",
# MAGIC   "- RETENTION PERIOD: ", coalesce(retention_years, 'Not Specified'), "\n",
# MAGIC   "- DATA QUALITY TIER: ", coalesce(data_quality_tier, 'Not Assessed'), "\n\n",
# MAGIC   
# MAGIC   "REGULATORY & COMPLIANCE:\n",
# MAGIC   "- COMPLIANCE FRAMEWORKS: ", coalesce(compliance_frameworks, 'Not Specified'), "\n",
# MAGIC   "- REGULATORY SCOPE: ", coalesce(regulatory_scope, 'Not Specified'), "\n\n",
# MAGIC   
# MAGIC   "GOVERNANCE ACCOUNTABILITY:\n",
# MAGIC   "- DATA STEWARD: ", coalesce(data_steward, 'Not Assigned'), "\n",
# MAGIC   "- STEWARD SQUAD: ", coalesce(steward_squad, 'Not Assigned'), "\n\n",
# MAGIC   
# MAGIC   "COMPLIANCE STATUS:\n",
# MAGIC   CASE 
# MAGIC     WHEN data_classification IS NOT NULL AND sensitivity IS NOT NULL 
# MAGIC     THEN "✓ CLASSIFICATION: Compliant\n"
# MAGIC     ELSE "✗ CLASSIFICATION: Non-Compliant - Missing Classifications\n"
# MAGIC   END,
# MAGIC   CASE 
# MAGIC     WHEN retention_policy IS NOT NULL OR retention_years IS NOT NULL
# MAGIC     THEN "✓ RETENTION: Policy Defined\n"
# MAGIC     ELSE "✗ RETENTION: Policy Missing\n" 
# MAGIC   END,
# MAGIC   CASE
# MAGIC     WHEN data_steward IS NOT NULL OR steward_squad IS NOT NULL
# MAGIC     THEN "✓ STEWARDSHIP: Assigned\n"
# MAGIC     ELSE "✗ STEWARDSHIP: Unassigned\n"
# MAGIC   END, "\n",
# MAGIC   
# MAGIC   "GOVERNANCE QUERIES (Natural Language):\n",
# MAGIC   "Q: Is ", schema_name, " confidential?\n",
# MAGIC   "A: ", CASE WHEN lower(coalesce(data_classification, '')) LIKE '%confidential%' 
# MAGIC     THEN 'YES - ' || data_classification ELSE 'NO' END, "\n\n",
# MAGIC   
# MAGIC   "Q: Is ", schema_name, " sensitive?\n",
# MAGIC   "A: ", CASE WHEN sensitivity IS NOT NULL OR sensitivity_level IS NOT NULL
# MAGIC     THEN 'YES - ' || coalesce(sensitivity, sensitivity_level) ELSE 'NO' END, "\n\n",
# MAGIC   
# MAGIC   "Q: Is ", schema_name, " high criticality?\n",
# MAGIC   "A: ", CASE WHEN lower(coalesce(criticality, data_criticality, '')) LIKE '%high%' 
# MAGIC          OR lower(coalesce(criticality, data_criticality, '')) LIKE '%tier 1%' 
# MAGIC     THEN 'YES - ' || coalesce(criticality, data_criticality) ELSE 'NO' END, "\n\n",
# MAGIC   
# MAGIC   "Q: What is the retention policy for ", schema_name, "?\n",
# MAGIC   "A: ", coalesce(retention_policy, retention_years, 'Not defined'), "\n\n",
# MAGIC   
# MAGIC   "Q: Does ", schema_name, " contain PII?\n",
# MAGIC   "A: ", coalesce(contains_pii, pii_content, 'Unknown'), "\n\n",
# MAGIC   
# MAGIC   "GOVERNANCE RELATIONSHIPS:\n",
# MAGIC   schema_name, " CLASSIFIED_AS ", coalesce(data_classification, 'unclassified'), "\n",
# MAGIC   schema_name, " SENSITIVITY_LEVEL ", coalesce(sensitivity, sensitivity_level, 'unknown'), "\n", 
# MAGIC   schema_name, " CRITICALITY_RATING ", coalesce(criticality, data_criticality, 'unassessed'), "\n",
# MAGIC   schema_name, " RETENTION_POLICY ", coalesce(retention_policy, 'undefined'), "\n",
# MAGIC   schema_name, " COMPLIANCE_SCOPE ", coalesce(compliance_frameworks, 'undefined'), "\n",
# MAGIC   schema_name, " REGULATED_BY ", coalesce(regulatory_scope, 'unknown'), "\n",
# MAGIC   coalesce(data_steward, 'unassigned'), " STEWARDS ", schema_name, "\n",
# MAGIC   coalesce(steward_squad, 'unassigned'), " ENSURES_COMPLIANCE ", schema_name, "\n\n",
# MAGIC   
# MAGIC   "KEYWORDS: ",
# MAGIC   "classification ", coalesce(data_classification, ''), " ",
# MAGIC   "confidential ", CASE WHEN lower(coalesce(data_classification, '')) LIKE '%confidential%' THEN 'yes' ELSE '' END, " ",
# MAGIC   "sensitivity ", coalesce(sensitivity, sensitivity_level, ''), " ",
# MAGIC   "sensitive ", coalesce(sensitivity, ''), " ",
# MAGIC   "criticality ", coalesce(criticality, data_criticality, ''), " ",
# MAGIC   "critical ", CASE WHEN lower(coalesce(criticality, '')) LIKE '%high%' THEN 'yes' ELSE '' END, " ",
# MAGIC   "retention ", coalesce(retention_policy, ''), " ",
# MAGIC   "compliance ", coalesce(compliance_frameworks, ''), " ",
# MAGIC   "regulatory ", coalesce(regulatory_scope, ''), " ",
# MAGIC   "governance ", coalesce(data_steward, ''), " ",
# MAGIC   "policy ", coalesce(retention_policy, ''), " ",
# MAGIC   "steward ", coalesce(data_steward, ''), " ",
# MAGIC   "pii ", coalesce(contains_pii, ''), " ",
# MAGIC   "encryption ", coalesce(encryption_required, ''), " "
# MAGIC ) AS document_text,
# MAGIC schema_name AS primary_key
# MAGIC FROM base;

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
vsc.delete_index(index_name="your_catalog.yourschema.metadata_identity_idx")
vsc.delete_index(index_name="your_catalog.yourschema.metadata_business_idx")
vsc.delete_index(index_name="your_catalog.yourschema.metadata_technical_idx")
vsc.delete_index(index_name="your_catalog.yourschema.metadata_governance_idx")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()

index = vsc.create_delta_sync_index(
    endpoint_name="one-env-shared-endpoint-1",
    source_table_name="your_catalog.yourschema.metadata_identity_source",
    index_name="your_catalog.yourschema.metadata_identity_idx",
    pipeline_type="TRIGGERED",
    primary_key="primary_key",
    embedding_source_column="document_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
print(f"Index created: {index}")


index = vsc.create_delta_sync_index(
    endpoint_name="one-env-shared-endpoint-1",
    source_table_name="your_catalog.yourschema.metadata_business_source",
    index_name="your_catalog.yourschema.metadata_business_idx",
    pipeline_type="TRIGGERED",
    primary_key="primary_key",
    embedding_source_column="document_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
print(f"Index created: {index}")


index = vsc.create_delta_sync_index(
    endpoint_name="one-env-shared-endpoint-1",
    source_table_name="your_catalog.yourschema.metadata_technical_source",
    index_name="your_catalog.yourschema.metadata_technical_idx",
    pipeline_type="TRIGGERED",
    primary_key="primary_key",
    embedding_source_column="document_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
print(f"Index created: {index}")


index = vsc.create_delta_sync_index(
    endpoint_name="one-env-shared-endpoint-1",
    source_table_name="your_catalog.yourschema.metadata_governance_source",
    index_name="your_catalog.yourschema.metadata_governance_idx",
    pipeline_type="TRIGGERED",
    primary_key="primary_key",
    embedding_source_column="document_text",
    embedding_model_endpoint_name="databricks-bge-large-en"
)
print(f"Index created: {index}")